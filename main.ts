// pumpfun-top5-forecast-01 entry v2: empirical rank-transition forecaster (Node 22, standard library only).
// A genuinely different method from the v1 Monte Carlo (momentum + crash mixture):
//   - it never simulates market caps. It watches how COINS MOVE BETWEEN RANKS.
//   - from every consecutive pair of 10-minute marks it counts rank-bucket
//     transitions (rank 1..10 as singletons, then 11-20, 21-50, 51-100, 101-200, OUT),
//     SPLIT BY VENUE (curve vs pool): a curve coin that spikes to rank 2 crashes
//     within a step, while a graduated pool coin at the same rank usually stays.
//     Without the split the matrix confuses the two and overrates newborns.
//     Counts are recency-weighted so the latest regime dominates, plus a Dirichlet
//     prior that keeps sparse rows sane;
//   - it composes the transition matrix over the horizon to each answer read (exact matrix
//     powers, no randomness), giving every coin a probability of holding each top-5 place;
//   - newborns (mints never seen before) enter through the OUT state at the empirical birth
//     rate and birth-rank distribution, as curve coins; their expected occupancy of a place
//     becomes that place's unnamed mass, which is what scores when a newcomer takes it;
//   - coins the model rates below ZERO_THRESHOLD at a place are named there at p = 0, which
//     takes them out of that place's unnamed share and concentrates it on the coins that
//     could actually hold it (including the newborns);
//   - sizes are a per-place mixture over the plausible holders: each holder
//     contributes a narrow normal around its current log-cap, and the newborn
//     share contributes a wide normal around the empirical top-5 newborn cap.
// Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.
// Deterministic: pure matrix arithmetic, no PRNG, no clock. Two runs are byte-identical.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MICRO = 1_000_000;
const MIN_P = Number(process.env.V2_MIN_P ?? 0.005); // rows below this are noise; dropping them feeds the unnamed mass
const MAX_ROWS_PER_PLACE = 100; // the rules' cap; named rows plus p = 0 rows
const ZERO_THRESHOLD = Number(process.env.V2_ZERO_THRESHOLD ?? 0.002); // name a coin at p = 0 when its modelled place p is below this
const AVGNEW_CAP = Number(process.env.V2_AVGNEW_CAP ?? 1.0); // cap on the newcomer share per place (1.0 = no cap)
const HALF_LIFE_PAIRS = Number(process.env.V2_HALF_LIFE ?? 2); // recency half-life of transition pairs, in pairs
const PRIOR_DIAG = Number(process.env.V2_PRIOR_DIAG ?? 1); // Dirichlet prior: stay in the same state
const PRIOR_ADJ = Number(process.env.V2_PRIOR_ADJ ?? 1); // ... move to a neighbouring state (one bucket, or a venue flip)
const PRIOR_REST = Number(process.env.V2_PRIOR_REST ?? 0.05); // ... move anywhere else
const OUT_STAY = 0.99; // a coin out of the chart stays out per step, unless reborn
const SIZE_COMP_SD = Number(process.env.V2_SIZE_COMP_SD ?? 0.15); // base sd of each holder's cap component
const SIZE_COMP_MULT = Number(process.env.V2_SIZE_COMP_MULT ?? 0.7); // multiplier on the venue sd's diffusion with horizon
const SIZE_NEW_SD = Number(process.env.V2_SIZE_NEW_SD ?? 0.5); // sd of the newborn component in the size mixture

// Rank buckets: 0..9 are ranks 1..10, 10 is 11-20, 11 is 21-50, 12 is 51-100,
// 13 is 101-200, 14 is OUT (not on the mark's rows at all).
// A state is (bucket, venue): venue 0 = curve (bonding curve), 1 = pool (graduated).
// States 0..27 are bucket*2+venue for buckets 0..13; state 28 is OUT (venue-free).
// Entry v2, submitted 12 Nov 2026 as the owner's final entry (replaces v1).
// Method: rank-transition forecaster — see README.md for the full description.
const N_BUCKETS = 15;
const OUT_BUCKET = 14;
const OUT = 28;
const N_STATES = 29;
function bucketOf(rank: number | null): number {
  if (rank === null) return OUT_BUCKET;
  if (rank <= 10) return rank - 1;
  if (rank <= 20) return 10;
  if (rank <= 50) return 11;
  if (rank <= 100) return 12;
  return 13;
}
const venueOf = (v: string): number => (v === 'pool' ? 1 : 0);
function stateOf(bucket: number, venue: number): number {
  return bucket === OUT_BUCKET ? OUT : bucket * 2 + venue;
}
function bucketOfState(s: number): number {
  return s === OUT ? OUT_BUCKET : Math.floor(s / 2);
}
function venueOfState(s: number): number {
  return s === OUT ? -1 : s % 2;
}
function adjacentState(a: number, b: number): boolean {
  if (a === OUT || b === OUT) return false;
  const ba = bucketOfState(a), bb = bucketOfState(b);
  const va = venueOfState(a), vb = venueOfState(b);
  return (Math.abs(ba - bb) === 1 && va === vb) || (ba === bb && va !== vb);
}

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  slot: number;
  name: string;
  symbol: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
  quote_reserves_lamports: string | number;
}
interface Mark {
  at: string;
  ok: boolean;
  rows: Row[] | null;
}

/** log10 of a cap in SOL, as the scorer takes it: log10(lamports) - 9. */
const sizeOf = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

/** erfc(x), Numerical Recipes' erfcc (fractional error under 1.2e-7 everywhere). */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
/** P(X >= x) for X normal with mean `mean` and sd `sd`. */
const above = (x: number, mean: number, sd: number): number => (x === -Infinity ? 1 : x === Infinity ? 0 : 0.5 * erfc((x - mean) / (sd * Math.SQRT2)));

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  size: { edges: number[] };
  reads: string[];
  cadence_minutes: number;
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const marks = lines
  .map((l) => JSON.parse(l) as Mark)
  .filter((m) => m.ok === true && Array.isArray(m.rows));
if (marks.length === 0) throw new Error('the history holds no ok mark');
const lastMark = marks[marks.length - 1]!;
const lastTime = new Date(lastMark.at).getTime();
const lastRows = [...(lastMark.rows ?? [])].sort((a, b) => a.rank - b.rank);
if (lastRows.length < task.places) throw new Error(`the last mark ranks ${lastRows.length} coins, fewer than the ${task.places} places`);
const cadenceMs = (task.cadence_minutes ?? 10) * 60000;

// ---- Fit the transition matrix from consecutive mark pairs (recency-weighted). ----
const counts: number[][] = Array.from({ length: N_STATES }, () => new Array<number>(N_STATES).fill(0));
const birthCounts: number[] = new Array<number>(N_STATES).fill(0); // entry states of never-seen mints
let birthWeightSum = 0;
let birthTotal = 0; // recency-weighted number of births per pair
const nPairs = Math.max(0, marks.length - 1);
const seenMints = new Set<string>();
// Seed `seen` with the first mark's mints so births are measured from the second mark on.
for (const r of marks[0]?.rows ?? []) seenMints.add(r.mint);
for (let j = 0; j < nPairs; j++) {
  const w = Math.pow(2, -(nPairs - 1 - j) / HALF_LIFE_PAIRS);
  const prev = new Map<string, { rank: number; venue: number }>();
  for (const r of marks[j]!.rows ?? []) prev.set(r.mint, { rank: r.rank, venue: venueOf(r.venue) });
  const cur = new Map<string, { rank: number; venue: number }>();
  for (const r of marks[j + 1]!.rows ?? []) cur.set(r.mint, { rank: r.rank, venue: venueOf(r.venue) });
  const mints = new Set([...prev.keys(), ...cur.keys()]);
  for (const mint of mints) {
    const p = prev.get(mint);
    const c = cur.get(mint);
    const from = stateOf(p ? bucketOf(p.rank) : OUT_BUCKET, p ? p.venue : 0);
    const to = stateOf(c ? bucketOf(c.rank) : OUT_BUCKET, c ? c.venue : 0);
    counts[from]![to]! += w;
  }
  // Births: mints at j+1 never seen in any mark up to j.
  let birthsThisPair = 0;
  for (const [mint, c] of cur) {
    if (!seenMints.has(mint)) {
      birthCounts[stateOf(bucketOf(c.rank), c.venue)]! += w;
      birthsThisPair++;
    }
  }
  birthWeightSum += w;
  birthTotal += w * birthsThisPair;
  for (const mint of cur.keys()) seenMints.add(mint);
}
const birthRate = birthWeightSum > 0 ? birthTotal / birthWeightSum : 0; // newborns per step
const birthDist: number[] = birthCounts.map((c) => (birthWeightSum > 0 ? c / birthWeightSum : 0));
{
  // Normalise the birth distribution over the states a newborn can enter.
  const s = birthDist.reduce((a, b) => a + b, 0);
  if (s > 0) for (let i = 0; i < N_STATES; i++) birthDist[i]! /= s;
  else {
    // No birth observed: newborns enter uniformly over the chart's states.
    for (let i = 0; i < N_STATES - 1; i++) birthDist[i]! = 1 / (N_STATES - 1);
  }
}

// Row-normalise counts + Dirichlet prior into the transition matrix T.
const T: number[][] = [];
for (let i = 0; i < N_STATES; i++) {
  const row: number[] = [];
  let sum = 0;
  for (let k = 0; k < N_STATES; k++) {
    const prior = i === k ? PRIOR_DIAG : adjacentState(i, k) ? PRIOR_ADJ : PRIOR_REST;
    const v = counts[i]![k]! + prior;
    row.push(v);
    sum += v;
  }
  T.push(row.map((v) => v / sum));
}
// The OUT row is unobservable in full (the chart never lists the coins that stay out):
// a coin out of the chart stays out with OUT_STAY, else it re-enters like a newborn.
for (let k = 0; k < N_STATES; k++) {
  T[OUT]![k] = k === OUT ? OUT_STAY : (1 - OUT_STAY) * birthDist[k]!;
}

/** One step of a row-vector distribution through T. */
function step(v: number[]): number[] {
  const out = new Array<number>(N_STATES).fill(0);
  for (let i = 0; i < N_STATES; i++) {
    if (v[i] === 0) continue;
    for (let k = 0; k < N_STATES; k++) out[k]! += v[i]! * T[i]![k]!;
  }
  return out;
}

// ---- Candidates: every coin on the last mark, with its current state. ----
interface Cand {
  mint: string;
  state: number;
}
const candList: Cand[] = lastRows.map((r) => ({ mint: r.mint, state: stateOf(bucketOf(r.rank), venueOf(r.venue)) }));

// ---- Sizes: per-coin current log-caps (for the holder mixture), the
// per-venue cap drift (a place-k cap is near its holder's PRESENT cap grown
// by the venue's drift over the horizon), and the per-place newborn log-cap
// (a newborn at place 1 is a much bigger coin than a newborn at place 5). ----
const lastLogCap = new Map<string, number>();
const lastVenue = new Map<string, number>();
for (const r of lastRows) {
  lastLogCap.set(r.mint, sizeOf(r.market_cap_lamports));
  lastVenue.set(r.mint, venueOf(r.venue));
}
const newCapsByPlace: number[][] = Array.from({ length: 5 }, () => []);
const newCapsAll: number[] = [];
{
  const seenSoFar = new Set<string>();
  for (const r of marks[0]?.rows ?? []) seenSoFar.add(r.mint);
  for (let j = 1; j < marks.length; j++) {
    for (const r of marks[j]!.rows ?? []) {
      if (!seenSoFar.has(r.mint) && r.rank <= 5) {
        newCapsByPlace[r.rank - 1]!.push(sizeOf(r.market_cap_lamports));
        newCapsAll.push(sizeOf(r.market_cap_lamports));
      }
    }
    for (const r of marks[j]!.rows ?? []) seenSoFar.add(r.mint);
  }
}
const newMuAll = newCapsAll.length > 0 ? newCapsAll.reduce((a, b) => a + b, 0) / newCapsAll.length : 2.9;
const newMuByPlace = newCapsByPlace.map((v) => (v.length > 0 ? v.reduce((a, b) => a + b, 0) / v.length : newMuAll));
// Per-venue per-step log-cap drift: the top-5 drift (the relevant one for a
// top-5 holder) shrunk toward the overall venue drift by sample size, so a
// handful of crashes cannot drag the estimate. Plus the per-step sd.
const venueDrift: number[] = [0, 0];
const venueSd: number[] = [0.12, 0.14];
{
  const overall: Array<{ n: number; sum: number; sq: number }> = [{ n: 0, sum: 0, sq: 0 }, { n: 0, sum: 0, sq: 0 }];
  const top5: Array<{ n: number; sum: number }> = [{ n: 0, sum: 0 }, { n: 0, sum: 0 }];
  for (let j = 0; j < nPairs; j++) {
    const prev = new Map<string, Row>();
    for (const r of marks[j]!.rows ?? []) prev.set(r.mint, r);
    for (const r of marks[j + 1]!.rows ?? []) {
      const p = prev.get(r.mint);
      if (!p) continue;
      const d = sizeOf(r.market_cap_lamports) - sizeOf(p.market_cap_lamports);
      const v = venueOf(p.venue);
      overall[v]!.n++;
      overall[v]!.sum += d;
      overall[v]!.sq += d * d;
      if (p.rank <= 5) {
        top5[v]!.n++;
        top5[v]!.sum += d;
      }
    }
  }
  for (let v = 0; v < 2; v++) {
    const o = overall[v]!;
    const t = top5[v]!;
    const oMean = o.n > 0 ? o.sum / o.n : 0;
    if (o.n > 1) venueSd[v] = Math.sqrt(Math.max(0, o.sq / o.n - oMean * oMean));
    const w = t.n / (t.n + 10); // shrink the noisy top-5 drift toward the overall drift
    venueDrift[v] = w * (t.n > 0 ? t.sum / t.n : oMean) + (1 - w) * oMean;
  }
}

// ---- Propagate to every answer read and average the place probabilities. ----
const reads = task.reads.map((s) => new Date(s).getTime());
// avgP[k][mint]: the read-averaged probability that mint holds place k+1.
// avgNew[k]: the read-averaged probability that a newborn holds place k+1.
const avgP: Array<Map<string, number>> = Array.from({ length: task.places }, () => new Map());
const avgNew: number[] = new Array<number>(task.places).fill(0);
for (const rt of reads) {
  const H = Math.max(0, (rt - lastTime) / cadenceMs);
  const steps = Math.round(H);
  // Each candidate's distribution after `steps` transitions.
  const dists = new Map<string, number[]>();
  for (const c of candList) {
    let v = new Array<number>(N_STATES).fill(0);
    v[c.state] = 1;
    for (let s = 0; s < steps; s++) v = step(v);
    dists.set(c.mint, v);
  }
  // The newborn pool: expected counts per state at the read. Each step the pool
  // evolves one transition and fresh births (birthRate, entry distribution) arrive.
  let pool = new Array<number>(N_STATES).fill(0);
  for (let s = 0; s < steps; s++) {
    pool = step(pool);
    for (let k = 0; k < N_STATES; k++) pool[k]! += birthRate * birthDist[k]!;
  }
  for (let k = 0; k < task.places; k++) {
    let sumRawRead = 0; // expected number of incumbent candidates at this place
    for (const c of candList) {
      // A coin holds place k+1 when its bucket is k (buckets 0..4 are ranks 1..5),
      // in either venue.
      const p = (dists.get(c.mint)![stateOf(k, 0)]! + dists.get(c.mint)![stateOf(k, 1)]!);
      sumRawRead += p;
      avgP[k]!.set(c.mint, (avgP[k]!.get(c.mint) ?? 0) + p / reads.length);
    }
    // The newborn share is a competing risk: the newborn stream (expected count
    // pool[k]) and the incumbents (expected count sumRawRead) contest the one
    // slot, so the newborn takes it with probability pool/(pool+incumbents).
    // This stops the pool from claiming every place when births are frequent:
    // a place with strong incumbents keeps most of its mass on them.
    const expectedNew = pool[stateOf(k, 0)]! + pool[stateOf(k, 1)]!;
    const denom = expectedNew + sumRawRead;
    avgNew[k]! += (denom > 0 ? expectedNew / denom : 0) / reads.length;
  }
}
const avgSteps = reads.reduce((a, rt) => a + Math.max(0, Math.round((rt - lastTime) / cadenceMs)), 0) / Math.max(1, reads.length);

if (process.env.V2_DEBUG) {
  console.error(`[v2] birthRate=${birthRate.toFixed(1)}/step drift=[${venueDrift.map((x) => x.toFixed(3)).join(',')}] sd=[${venueSd.map((x) => x.toFixed(3)).join(',')}] newMuByPlace=[${newMuByPlace.map((x) => x.toFixed(2)).join(',')}] avgSteps=${avgSteps.toFixed(1)} avgNew=[${avgNew.map((x) => x.toFixed(2)).join(',')}]`);
}

// ---- Assemble the forecast: normalise each place, name the likely, zero the hopeless. ----
const edges = task.size.edges;
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
const sizes: Array<{ place: number; bin: number; p: number }> = [];
const top5: string[] = [];
const chosen = new Set<string>();
for (let k = 0; k < task.places; k++) {
  const raw = [...avgP[k]!.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  const newcomer = Math.max(0, Math.min(AVGNEW_CAP, avgNew[k]!));
  const sumRaw = raw.reduce((a, [, p]) => a + p, 0);
  // Scale the existing coins so they hold exactly (1 - newcomer) of the place:
  // one coin holds each place, so the named and the unnamed must add to 1.
  const scale = sumRaw > 0 ? (1 - newcomer) / sumRaw : 0;
  const scaled = raw.map(([mint, p]) => [mint, p * scale] as const);
  let rowsUsed = 0;
  for (const [mint, p] of scaled) {
    if (rowsUsed >= MAX_ROWS_PER_PLACE) break;
    if (p >= MIN_P) {
      probabilities.push({ mint, place: k + 1, p: Math.floor(p * MICRO) / MICRO });
      rowsUsed++;
    }
  }
  // Zero out the least likely coins (up to the row cap): they leave the unnamed
  // share to the coins that could actually hold the place, newborns among them.
  const tail = [...scaled].sort((a, b) => a[1] - b[1]);
  for (const [mint, p] of tail) {
    if (rowsUsed >= MAX_ROWS_PER_PLACE) break;
    if (p >= MIN_P) continue; // already named above
    if (p < ZERO_THRESHOLD) {
      probabilities.push({ mint, place: k + 1, p: 0 });
      rowsUsed++;
    }
  }
  // Modal top 5, distinct mints, by the scaled place probability.
  for (const [mint] of scaled) {
    if (!chosen.has(mint)) {
      top5.push(mint);
      chosen.add(mint);
      break;
    }
  }
  // Size distribution: a mixture over the plausible holders. Each holder
  // contributes a normal around its PROJECTED log-cap (its current cap grown
  // by its venue's drift over the mean horizon, sd diffusing with the horizon);
  // the newborn share contributes a wide normal around the empirical log-cap
  // of a newborn that reaches THIS place.
  // Two guards keep the mixture honest: it is built only from the holders
  // that actually matter at this place (p >= 0.01), and it is truncated at
  // log 2.00 (100 SOL) and renormalised — a top-5 coin below 100 SOL
  // contradicts the chart's own scale (the #5 coin is ~500 SOL), so the
  // normals' left tails below 100 SOL are wasted mass, not real uncertainty.
  const comps: Array<[number, number, number]> = [];
  for (const [mint, p] of scaled) {
    if (p < 0.01 || !lastLogCap.has(mint)) continue;
    const v = lastVenue.get(mint) ?? 0;
    const mu = lastLogCap.get(mint)! + venueDrift[v]! * avgSteps;
    const sd = Math.min(0.5, SIZE_COMP_SD + SIZE_COMP_MULT * venueSd[v]! * Math.sqrt(Math.max(0, avgSteps)));
    comps.push([mu, sd, p]);
  }
  comps.push([newMuByPlace[k]!, SIZE_NEW_SD, newcomer]);
  const binP = new Array<number>(edges.length + 1).fill(0);
  for (let bin = 0; bin <= edges.length; bin++) {
    const lo = bin === 0 ? -Infinity : edges[bin - 1]!;
    const hi = bin === edges.length ? Infinity : edges[bin]!;
    // Truncate at log 2.00: bin 0 (below 100 SOL) gets no mass.
    if (hi <= 2.0) continue;
    const loEff = lo === -Infinity || lo < 2.0 ? 2.0 : lo;
    let mass = 0;
    for (const [mu, sd, w] of comps) mass += w * (above(loEff, mu, sd) - above(hi, mu, sd));
    binP[bin] = mass;
  }
  const tot = binP.reduce((a, b) => a + b, 0);
  if (tot > 0) {
    for (let bin = 0; bin < binP.length; bin++) {
      const u = Math.floor((binP[bin]! / tot) * MICRO);
      if (u > 0) sizes.push({ place: k + 1, bin, p: u / MICRO });
    }
  }
}
if (top5.length !== task.places) throw new Error(`top5 holds ${top5.length} coins, expected ${task.places}`);
if (new Set(top5).size !== top5.length) throw new Error('top5 holds a duplicate mint');

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);

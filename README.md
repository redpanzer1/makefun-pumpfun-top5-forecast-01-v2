# pumpfun-top5-forecast-01 — entry v2 (rank-transition forecaster)

Second entry for `pumpfun-top5-forecast-01`, submitted 12 Nov 2026 by the
owner's decision after the v1 entry. Per the competition rules the last
submission before the deadline is the final entry, so this v2 **replaces**
the v1 Monte Carlo entry.

## Method

A genuinely different method from v1 (Monte Carlo market-cap simulation).
This entry **never simulates market caps** — it watches how coins move
between ranks:

- **Rank-transition (Markov) matrix.** From every consecutive pair of
  10-minute marks it counts transitions between rank buckets (ranks 1–10
  as singletons, then 11–20, 21–50, 51–100, 101–200, OUT), **split by
  venue (curve vs pool)**. A curve coin that spikes to rank 2 crashes
  within a step, while a graduated pool coin at the same rank usually
  stays. Counts are recency-weighted (half-life 2 pairs) with a Dirichlet
  prior. Exact matrix powers give each coin's probability of holding each
  top-5 place at each answer read — deterministic, no PRNG, no clock.
- **Newborns** enter through the OUT state at the empirical birth rate and
  birth-rank distribution, as curve coins. Their expected occupancy of a
  place becomes that place's unnamed mass (competing-risk share).
- **Zero threshold:** coins rated below 0.002 at a place are named there at
  p = 0, concentrating the unnamed share on coins that could actually
  hold the place.
- **Sizes:** per-place mixture over plausible holders — a narrow normal
  around each holder's current log-cap, plus a wide newborn component
  around the empirical top-5 newborn cap.

## Backtest (6 practice reads, higher is better, 0 best)

| Entry | order | size | total |
|---|---|---|---|
| v1 (Monte Carlo, replaced) | −19.32 | −15.36 | **−34.68** |
| **v2 (this entry)** | −20.38 | −18.02 | **−38.40** |
| Starter (persistence) | −21.86 | −19.25 | −41.12 |

V2 beats the starter (+2.72) but trails v1 (−3.72) on the 6-read practice
backtest. It was submitted anyway on the owner's explicit instruction
("Gas aja", 12 Nov 2026 13:34 WIB), as a diversified hedge: the transition
matrix is estimated from only 4 history pairs in practice but from 65
pairs in the final, so the backtest likely underestimates it.

## Reproduce

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Two runs on the same input produce byte-identical `forecast.json`.
Node 22, standard library only, no network.

Submission: repo `makefun-pumpfun-top5-forecast-01-v2`, owner wallet in
`.makefun/owner.json`, MIT license.

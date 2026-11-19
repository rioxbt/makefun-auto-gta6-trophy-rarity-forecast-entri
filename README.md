# makefun-auto-gta6-trophy-rarity-forecast-entri

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rioxbt/makefun-auto-gta6-trophy-rarity-forecast-entri/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rioxbt/makefun-auto-gta6-trophy-rarity-forecast-entri/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A five-quantile forecaster of Grand Theft Auto VI's PlayStation 5 trophy earn rates for
[make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).
It reads one case — every reading the house made of Exophase's GTA VI trophy page before a checkpoint — and writes,
for each of the game's trophies, the 5%, 25%, 50%, 75% and 95% quantiles of that trophy's earn rate at the case's
target. It runs offline, on Node's standard library alone, and it is deterministic: two runs on one case write the
same bytes.

## How it forecasts

Trophy earn rates only change when Exophase refreshes a game's page; between refreshes the page is frozen (launch
night printed 638 players tracked and the same rate for every trophy from 23:48Z until the 04:30Z refresh). So each
trophy's informative history is the sequence of reads where its rate changed — the level a refresh set, and when it
was first seen.

1. **Level.** A rate is modelled on the logit scale as a straight line in `x = ln(1 + hours since release)`, i.e. the
   rate rises like a saturating adoption curve, logistic in time. Each trophy's line is fitted to its own step points
   by recency-weighted least squares; a trophy with too few points, and part of every trophy's slope, borrows the
   weighted median slope across all the trophies, because the trophies share one launch curve and pooling its shape is
   what 55 short series are good for.
2. **Drift.** The line is anchored on the last observed level — the surest number in the case — and read at the target,
   with the extrapolated increment damped, because adoption decelerates and the target sits well beyond the case's own
   window.
3. **Spread.** The same predictor is re-run as a walk-forward backtest over the case's own refresh instants: fit on the
   points known at an instant, forecast a later instant, record the logit error. Those errors give the growth of the
   uncertainty with lead time (a power law, read at the case's horizon) and the shape of the residual error
   distribution (its 5/25/50/75/95% quantiles in units of that spread), so the five quantiles carry the skew and fat
   tails the game's own refresh jumps produced instead of assuming a normal. A trophy whose own backtest errors are
   wide gets a wider band, a floor keeps the band honest when a case is too quiet to measure, and a case whose whole
   history shows the page never changing — the refresh process entirely unobserved — leans a little low and a little
   wide, since the profiles that sync before the target dilute the rates.
4. **Fallbacks.** Every row is clamped to `[0, 100]`, rounded to 4 decimals and made non-decreasing, and a case that
   cannot be solved at all falls back to the persistence baseline's numbers, so a valid forecast is always written.

`MAKEFUN_DEBUG=1` prints one diagnostic line a case on stderr (refresh instants, backtest errors, measured and used
spreads, the error shape); a graded run sets nothing and stays silent.

## Run it

The grader runs one case at a time: `<id>.in` in `/input`, the forecast written to `/output/<id>.out`.

```sh
mkdir -p input out
cp <the kit>/input/*.in input/
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

On Node 22, which strips types only behind a flag:

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node --experimental-strip-types main.ts
```

## What an entry must satisfy

- Entry point is `main.ts` at the repository root, Node 22, standard library only, no network, nothing installed.
- It reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `$MAKEFUN_OUTPUT_DIR/<id>.out`
  (default `/output`), each at most 65,536 bytes of UTF-8 JSON: `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]},...]}`,
  one row per trophy, no other key, five finite non-decreasing numbers from 0 to 100.
- Deterministic, and far inside the 8-second, 1-CPU, 1024 MB cap: one 55-trophy, 300-read case takes about 0.2 s.

## License

MIT — see [LICENSE](LICENSE).

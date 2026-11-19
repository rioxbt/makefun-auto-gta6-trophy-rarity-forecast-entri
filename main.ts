#!/usr/bin/env node
// gta6-trophy-rarity-forecast-01 — quantile forecaster for GTA VI PS5 trophy earn rates (Node 22, standard library
// only, offline). It reads every <id>.in case in the input directory and writes one <id>.out forecast per case: for
// each of the case's trophies, the five quantiles of its earn rate at the case's target, at the levels 0.05, 0.25,
// 0.5, 0.75 and 0.95, scored by a pinball loss on the logit scale. Nothing is fetched and nothing is random, so two
// runs on one case write the same bytes.
//
// What the program does with one case (all of it from the case file alone):
//
//   1. The house's reads only change when Exophase refreshes the page: the rates are frozen between refreshes
//      (launch night kept 638 players tracked and every rate the same from 23:48Z to 04:00Z). So a trophy's
//      informative points are the reads where its rate changed, each carrying the level Exophase last refreshed it
//      to and the moment that level was first seen.
//   2. A rate's level is modelled on the logit scale as a straight line in x = ln(1 + hours since release):
//      logit(r) = a + b x, i.e. r rises like a saturating adoption curve (logistic in time). Each trophy's line is
//      fitted to its own points by recency-weighted least squares; a trophy with too few points, and part of every
//      trophy's slope, borrows the weighted median slope of all the trophies, because the trophies share one launch
//      curve and pooling its shape is what 55 short series are good for. The line is anchored on the last observed
//      level — the surest number in the case — and read at the target with the extrapolated increment damped, since
//      adoption decelerates.
//   3. The spread is calibrated on the case itself. The same predictor is re-run as a walk-forward backtest over the
//      case's own refresh instants: fit on the points known at an instant, forecast a later instant, record the
//      logit error. Those errors give the growth of the uncertainty with lead time (a power law sd ~ lead^p, read at
//      the case's horizon) and the shape of the residual error distribution (its 5/25/50/75/95% quantiles in units
//      of that sd), so the five quantiles carry the skew and fat tails the game's own refresh jumps produced instead
//      of assuming a normal. A trophy whose own backtest errors are wide gets a wider band; a floor keeps the band
//      honest when the case is too quiet to measure; and a case whose whole history shows the page never changing —
//      so the refresh process behind the rates is entirely unobserved — leans that little bit low and wide, since the
//      profiles that sync before the target dilute the rates.
//   4. Every row is clamped to [0, 100], rounded to 4 decimals and made non-decreasing, and a case that cannot be
//      solved at all falls back to the persistence baseline's numbers, so a valid forecast is always written.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level: the shape used when the case is too quiet to measure its own. */
const GAUSS_Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** A rate is clamped to this window before its logit, exactly as the scorer clamps it. */
const RATE_FLOOR = 0.01;
const RATE_CEIL = 99.99;
/** The persistence baseline's spread, kept for the fallback path: logit units per square-root day. */
const BASELINE_SIGMA = 0.5;
/** The recency decay of a series' point weights: each earlier point counts this much less than the next. */
const POINT_DECAY = 0.8;
/** How much of a trophy's own fitted slope to trust against the pooled slope of all the trophies. */
const OWN_SLOPE_WEIGHT = 0.7;
/** The damping of the extrapolated increment: adoption decelerates, and the target lies well beyond the case's own
 *  window, so the line's full rise is not taken. */
const DRIFT_DAMPING = 0.6;
/** The largest drift the model forecasts, in logit units, and the largest bias the backtest may correct. */
const MAX_DRIFT = 2.5;
const MAX_BIAS = 0.7;
/** The backtest's spread is deflated: it is measured with less history than the live fit has. */
const BACKTEST_DEFLATION = 0.85;
/** The floor of the spread, in logit units: 0.10 + 0.30 * sqrt(horizon in days). */
const FLOOR_SIGMA_BASE = 0.1;
const FLOOR_SIGMA_PER_ROOT_DAY = 0.3;
/** The spread is clamped to this window, and a trophy's own backtest scales it by at most this factor. */
const SIGMA_MIN = 0.08;
const SIGMA_MAX = 2.6;
const SCALE_MIN = 0.7;
const SCALE_MAX = 1.6;
const SCALE_SHRINK = 0.5;
/** A trophy needs at least this many of its own backtest errors before it scales its own band. */
const MIN_OWN_ERRORS = 3;
/** The backtest needs at least this many usable errors, else the spread leans entirely on the floor. */
const MIN_ERRORS = 24;
/** A backtest cut fits only the points a trophy has then, and a cut too early would fit a line to rubble: no cut
 *  before this share of the history's span, and no trophy counted at a cut that has fewer than this many points. */
const CUT_EARLIEST_SHARE = 0.35;
const MIN_FIT_POINTS = 5;
/** The empirical shape's tails are clipped to this, so a handful of early errors cannot open the band to nothing. */
const MAX_SHAPE_Z = 2.2;
/** An error is measured in units of the spread fitted at its own lead, but never against a smaller spread than this:
 *  a power law read at a short lead can be near zero, and dividing by it would turn rounding noise into fat tails. */
const SHAPE_SD_FLOOR = 0.3;
/** A case whose whole history shows one unchanged page gives no evidence at all about the site's refresh process: the
 *  last rate is as of a refresh the case never saw, and the profiles that sync between it and the target are newer and
 *  less far in, which dilutes the rates. Such a case leans that little bit low and widens, rather than trusting a
 *  frozen page for a week. The hedge is inert on any case whose history shows the page change even a few times. */
const STALE_PAGE_STATES = 3;
const STALE_SHIFT = 0.4;
const STALE_WIDEN = 0.5;
/** How much of the backtest's median error to carry into the forecast's median: a bias is real but its size is not. */
const BIAS_DAMPING = 0.8;
/** The lowest x a moment may map to, so ln(1 + hours) stays finite at and before release. */
const MIN_HOURS = 0.25;
const MAX_CUTS = 24;
const MAX_EVALS_PER_CUT = 6;

const HOUR_MS = 3_600_000;
const MINUTE_MS = 60_000;

type Rates = Array<number | null>;

interface HistoryRead {
  slot?: string;
  at?: string;
  players_tracked?: number;
  trophies_earned?: number;
  rates?: Rates;
}

interface TrophyCase {
  schema?: string;
  id?: string;
  at?: string;
  horizon_minutes?: number;
  target?: string;
  levels?: number[];
  game?: { name?: string; platform?: string; release_date?: string; source?: string };
  trophies?: Array<{ id?: number; tier?: string; name?: string }>;
  history?: HistoryRead[];
}

/** One trophy's informative point: the moment a read first showed it at a new rate. */
interface Step {
  t: number;
  r: number;
}

/** A recency-weighted least-squares line of logit rates on x, with how firmly the points fix its slope. */
interface Line {
  b: number;
  sxx: number;
}

/** One backtest error: how far the live predictor was off, `lead` hours ahead of the cut it was fitted on. */
interface ErrorPair {
  lead: number;
  e: number;
  trophy: number;
}

const clamp = (x: number, lo: number, hi: number): number => (x < lo ? lo : x > hi ? hi : x);
/** A rate clamped to [0.01, 99.99], as the scorer clamps it. */
const clampRate = (r: number): number => clamp(r, RATE_FLOOR, RATE_CEIL);
/** The logit of a rate in percent, clamped first: ln(c / (100 - c)). */
const logit = (r: number): number => {
  const c = clampRate(r);
  return Math.log(c / (100 - c));
};
/** The rate in percent whose logit is l: the scorer's inverse, 100 / (1 + exp(-l)). */
const expit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const mean = (xs: number[]): number => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);

/** A moment in ms from an ISO 8601 string, or null. */
function timeOf(text: string | undefined): number | null {
  if (typeof text !== 'string' || text === '') return null;
  const ms = Date.parse(text);
  return Number.isFinite(ms) ? ms : null;
}

/** Recency weights over n points: the newest is 1, each earlier one POINT_DECAY less than the next. */
function weightsOf(n: number): number[] {
  const w: number[] = new Array(n);
  let v = 1;
  for (let k = n - 1; k >= 0; k--) {
    w[k] = v;
    v *= POINT_DECAY;
  }
  return w;
}

/** The recency-weighted least-squares slope of ys on xs (0 when the xs do not vary). */
function wls(xs: number[], ys: number[]): Line {
  const n = xs.length;
  const ws = weightsOf(n);
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    const w = ws[i] as number;
    sw += w;
    sx += w * (xs[i] as number);
    sy += w * (ys[i] as number);
  }
  if (!(sw > 0)) return { b: 0, sxx: 0 };
  const xb = sx / sw;
  const yb = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const w = ws[i] as number;
    const dx = (xs[i] as number) - xb;
    sxx += w * dx * dx;
    sxy += w * dx * ((ys[i] as number) - yb);
  }
  return { b: sxx > 1e-12 ? sxy / sxx : 0, sxx };
}

/** The weighted median of vals under wts (same length, non-negative). */
function weightedMedian(vals: number[], wts: number[]): number {
  const order = vals.map((_, i) => i).sort((a, b) => (vals[a] as number) - (vals[b] as number));
  let total = 0;
  for (const w of wts) total += w;
  let acc = 0;
  for (const i of order) {
    acc += wts[i] as number;
    if (acc >= total / 2) return vals[i] as number;
  }
  return vals[order[order.length - 1] as number] as number;
}

/** The empirical quantile at p of an ascending array. */
function quantile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0] as number;
  const pos = p * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(sorted.length - 1, lo + 1);
  const frac = pos - lo;
  return (sorted[lo] as number) * (1 - frac) + (sorted[hi] as number) * frac;
}

/** Everything one case's forecast needs, built once from its history. */
interface Prepared {
  /** The case's tropies' step points, ascending, one series a trophy in the case's order. */
  steps: Step[][];
  /** The refresh instants: the moments a read first showed a changed page (the first read included). */
  refreshTimes: number[];
  /** x = ln(1 + hours since release) at a moment. */
  xOf: (t: number) => number;
}

/** The release instant the case's clock runs from, or the best stand-in the case allows. */
function releaseTimeOf(c: TrophyCase, times: number[]): number {
  const date = c.game?.release_date;
  if (typeof date === 'string' && date.length >= 10) {
    const ms = Date.parse(`${date.slice(0, 10)}T00:00:00Z`);
    if (Number.isFinite(ms)) return ms;
  }
  if (times.length > 0) return (times[0] as number) - 12 * HOUR_MS;
  const at = timeOf(c.at);
  return at !== null ? at - 12 * HOUR_MS : 0;
}

/** Whether two reads show a different page: any rate, or either header count. */
function differs(a: HistoryRead, b: HistoryRead, nT: number): boolean {
  if (a.players_tracked !== b.players_tracked || a.trophies_earned !== b.trophies_earned) return true;
  const ra = Array.isArray(a.rates) ? a.rates : [];
  const rb = Array.isArray(b.rates) ? b.rates : [];
  const n = Math.min(nT, ra.length, rb.length);
  for (let i = 0; i < n; i++) if (ra[i] !== rb[i]) return true;
  return false;
}

function prepare(c: TrophyCase): Prepared {
  const history = Array.isArray(c.history) ? c.history : [];
  const trophies = Array.isArray(c.trophies) ? c.trophies : [];
  const nT = trophies.length;

  const times: number[] = [];
  const refreshTimes: number[] = [];
  let previous: HistoryRead | null = null;
  for (let i = 0; i < history.length; i++) {
    const read = history[i] as HistoryRead;
    const t = timeOf(read.at) ?? timeOf(read.slot);
    if (t === null) continue;
    times.push(t);
    if (previous === null || differs(previous, read, nT)) refreshTimes.push(t);
    previous = read;
  }

  const release = releaseTimeOf(c, times);
  const xOf = (t: number): number => Math.log(1 + Math.max(MIN_HOURS, (t - release) / HOUR_MS));

  const steps: Step[][] = [];
  for (let i = 0; i < nT; i++) {
    const own: Step[] = [];
    let last: number | null = null;
    for (const read of history) {
      const t = timeOf(read.at) ?? timeOf(read.slot);
      if (t === null) continue;
      const rates = Array.isArray(read.rates) ? read.rates : [];
      const r = rates[i];
      if (!finite(r)) continue;
      if (last === null || r !== last) {
        own.push({ t, r });
        last = r;
      }
    }
    steps.push(own);
  }
  return { steps, refreshTimes, xOf };
}

/** The number of a trophy's steps at or before t. */
function prefixAt(steps: Step[], t: number): number {
  let lo = 0;
  let hi = steps.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((steps[mid] as Step).t <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** The step a trophy's series shows at t: its last step at or before t, else its first. */
function stepAt(steps: Step[], t: number): Step | null {
  const k = prefixAt(steps, t);
  if (k > 0) return steps[k - 1] as Step;
  return steps.length > 0 ? (steps[0] as Step) : null;
}

/** The slope a trophy's known points give it: its own, shrunk toward the pooled slope of all the trophies. */
function slopeOf(points: Step[], pooled: number, xOf: (t: number) => number): number {
  const n = points.length;
  if (n < 2) return pooled;
  const own = wls(points.map((p) => xOf(p.t)), points.map((p) => logit(p.r))).b;
  if (n < 3) return 0.5 * own + 0.5 * pooled;
  return OWN_SLOPE_WEIGHT * own + (1 - OWN_SLOPE_WEIGHT) * pooled;
}

/** The logit the predictor reads at `at` from a trophy's known points. */
function predictLogit(points: Step[], pooled: number, xOf: (t: number) => number, at: number): number | null {
  const n = points.length;
  if (n === 0) return null;
  const last = points[n - 1] as Step;
  const lLast = logit(last.r);
  if (n < 2) return lLast + clamp(DRIFT_DAMPING * pooled * (xOf(at) - xOf(last.t)), -MAX_DRIFT, MAX_DRIFT);
  const b = slopeOf(points, pooled, xOf);
  const drift = clamp(DRIFT_DAMPING * b * (xOf(at) - xOf(last.t)), -MAX_DRIFT, MAX_DRIFT);
  return lLast + drift;
}

/** The pooled slope of every trophy with at least three points, as the weighted median of their own slopes. */
function pooledSlopeOf(steps: Step[][], xOf: (t: number) => number): number {
  const vals: number[] = [];
  const wts: number[] = [];
  for (const own of steps) {
    if (own.length < 3) continue;
    const line = wls(own.map((p) => xOf(p.t)), own.map((p) => logit(p.r)));
    vals.push(line.b);
    wts.push(Math.max(1e-9, line.sxx));
  }
  return vals.length > 0 ? weightedMedian(vals, wts) : 0;
}

/**
 * The case's walk-forward backtest: the live predictor refitted on the points known at a refresh instant and read
 * at later instants, its logit error recorded with the lead time. It is the evidence the spread and its shape rest
 * on. Cuts live in the later part of the history (a cut too early would fit a line to three points and pay for it),
 * so the errors measure the model the case is really forecast with, and the lead dependence is extrapolated from there.
 */
function backtest(prep: Prepared, nT: number): ErrorPair[] {
  const instants = prep.refreshTimes;
  const out: ErrorPair[] = [];
  if (instants.length < 3) return out;

  const span = (instants[instants.length - 1] as number) - (instants[0] as number);
  const firstCut = instants.findIndex((t) => t - (instants[0] as number) >= CUT_EARLIEST_SHARE * span);
  const last = instants.length - 2;
  if (firstCut < 0 || firstCut > last) return out;

  const cutCount = Math.min(MAX_CUTS, last - firstCut + 1);
  for (let k = 0; k < cutCount; k++) {
    const ci = firstCut + Math.floor((k * (last - firstCut)) / Math.max(1, cutCount - 1));
    const tCut = instants[ci] as number;
    // The points every trophy has at this cut, once: the pooled slope and each trophy's own slope come from them.
    const known = prep.steps.map((own) => own.slice(0, prefixAt(own, tCut)));
    const pooled = pooledSlopeOf(known, prep.xOf);
    const fitted = known.map((points) => (points.length >= MIN_FIT_POINTS ? slopeOf(points, pooled, prep.xOf) : null));

    const rest = instants.length - 1 - ci;
    const want = Math.min(MAX_EVALS_PER_CUT, rest);
    const evals: number[] = [];
    for (let m = 0; m < want; m++) {
      const t = instants[ci + 1 + Math.floor((m * rest) / want)] as number;
      if (!evals.includes(t)) evals.push(t);
    }

    for (let i = 0; i < nT; i++) {
      const b = fitted[i];
      const points = known[i] as Step[];
      if (b === null || points.length === 0) continue;
      const lastPoint = points[points.length - 1] as Step;
      const lLast = logit(lastPoint.r);
      for (const tEval of evals) {
        const seen = stepAt(prep.steps[i] as Step[], tEval);
        if (seen === null) continue;
        const drift = clamp(DRIFT_DAMPING * b * (prep.xOf(tEval) - prep.xOf(lastPoint.t)), -MAX_DRIFT, MAX_DRIFT);
        out.push({ lead: (tEval - tCut) / HOUR_MS, e: logit(seen.r) - (lLast + drift), trophy: i });
      }
    }
  }
  return out;
}

/** The fit of the backtest errors' size: sd(lead) = A * lead^p, or null when there is too little to fit. */
function sizeFit(errors: ErrorPair[]): { lnA: number; p: number } | null {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const { lead, e } of errors) {
    const a = Math.abs(e);
    if (lead <= 0 || !(a > 1e-9)) continue;
    xs.push(Math.log(lead));
    ys.push(Math.log(a));
  }
  if (xs.length < MIN_ERRORS) return null;
  const xb = mean(xs);
  const yb = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = (xs[i] as number) - xb;
    sxx += dx * dx;
    sxy += dx * ((ys[i] as number) - yb);
  }
  if (!(sxx > 1e-12)) return null;
  // E[ln|e|] = ln(sd) + E[ln|Z|] with E[ln|Z|] = -(gamma + ln 2)/2 = -0.6352, so undo it in lnA.
  return { lnA: yb - clamp(sxy / sxx, 0.15, 0.85) * xb + 0.6352, p: clamp(sxy / sxx, 0.15, 0.85) };
}

/** The sd in logit units that a fit reads at a lead in hours, or null. */
function sdAt(fit: { lnA: number; p: number } | null, lead: number): number | null {
  if (fit === null) return null;
  const sd = Math.exp(fit.lnA) * Math.pow(Math.max(lead, 1e-6), fit.p);
  return Number.isFinite(sd) && sd > 0 ? sd : null;
}

/** The forecast: the five quantile rows of one case. */
function solve(c: TrophyCase): Array<{ id: number; q: number[] }> {
  const trophies = Array.isArray(c.trophies) ? c.trophies : [];
  const nT = trophies.length;
  if (nT === 0) return [];
  const ids = trophies.map((t, i) => (finite(t.id) ? (t.id as number) : i + 1));
  const horizonMinutes = finite(c.horizon_minutes) ? Math.max(0, c.horizon_minutes as number) : 0;

  const prep = prepare(c);
  const lastRate = (i: number): number => {
    const own = prep.steps[i];
    return own !== undefined && own.length > 0 ? (own[own.length - 1] as Step).r : RATE_FLOOR;
  };
  const fallback = (): Array<{ id: number; q: number[] }> =>
    ids.map((id, i) => baselineRow(id, lastRate(i), horizonMinutes));

  const targetMs = timeOf(c.target) ?? null;
  const atMs = timeOf(c.at) ?? lastReadTime(c) ?? targetMs;
  // A case that names no target still names a horizon: the target is the checkpoint plus it.
  const resolvedTarget = targetMs ?? (atMs !== null ? atMs + horizonMinutes * MINUTE_MS : null);
  if (prep.steps.every((own) => own.length === 0) || resolvedTarget === null || atMs === null) return fallback();

  const horizonHours = Math.max(0, (resolvedTarget - atMs) / HOUR_MS);
  if (!(horizonHours > 0)) return fallback();

  const errors = backtest(prep, nT);
  const fit = sizeFit(errors);
  const floorSigma = FLOOR_SIGMA_BASE + FLOOR_SIGMA_PER_ROOT_DAY * Math.sqrt(horizonHours / 24);
  const measured = sdAt(fit, horizonHours);
  const baseSigma = clamp(
    Math.max(measured === null ? 0 : measured * BACKTEST_DEFLATION, floorSigma),
    SIGMA_MIN,
    SIGMA_MAX,
  );

  // The shape of the residual error, in units of the fitted sd, and each trophy's own error size.
  const zs: number[] = [];
  const perTrophy = new Map<number, number[]>();
  for (const { lead, e, trophy } of errors) {
    const sd = sdAt(fit, lead);
    if (sd === null) continue;
    const z = clamp(e / Math.max(sd, SHAPE_SD_FLOOR), -6, 6);
    zs.push(z);
    const list = perTrophy.get(trophy) ?? [];
    list.push(z);
    perTrophy.set(trophy, list);
  }
  let shape = [...GAUSS_Z];
  if (zs.length >= MIN_ERRORS) {
    zs.sort((a, b) => a - b);
    const blend = clamp(zs.length / (4 * MIN_ERRORS), 0, 1);
    shape = LEVELS.map((tau, j) =>
      clamp(blend * quantile(zs, tau) + (1 - blend) * (GAUSS_Z[j] as number), -MAX_SHAPE_Z, MAX_SHAPE_Z),
    );
    // The median carries the backtest's verdict on the model's bias, damped: a bias is real, its measured size is noisy.
    shape[2] = clamp((shape[2] as number) * BIAS_DAMPING, -MAX_BIAS, MAX_BIAS);
    for (let j = 1; j < shape.length; j++) {
      if ((shape[j] as number) < (shape[j - 1] as number)) shape[j] = shape[j - 1] as number;
    }
  }
  if (process.env.MAKEFUN_DEBUG === '1') {
    note(
      `${c.id ?? '?'}: ${prep.steps.length} trophies, ${prep.refreshTimes.length} refresh instants, ` +
        `${errors.length} backtest errors, horizon ${horizonHours.toFixed(1)}h, ` +
        `measured sd ${measured === null ? '-' : (measured * BACKTEST_DEFLATION).toFixed(3)}, ` +
        `floor ${floorSigma.toFixed(3)}, base sigma ${baseSigma.toFixed(3)}, shape ${shape.map((z) => z.toFixed(3)).join(',')}`,
    );
  }

  const pooled = pooledSlopeOf(prep.steps, prep.xOf);
  // A case whose history never shows the page change: hedge, see STALE_PAGE_STATES.
  const stale = prep.refreshTimes.length < STALE_PAGE_STATES;

  return ids.map((id, i) => {
    const own = prep.steps[i] as Step[];
    if (own.length === 0) return baselineRow(id, RATE_FLOOR, horizonMinutes);
    const predicted = predictLogit(own, pooled, prep.xOf, resolvedTarget);
    const l = clamp(
      (predicted === null ? logit(lastRate(i)) : predicted) - (stale ? STALE_SHIFT : 0),
      logit(RATE_FLOOR),
      logit(RATE_CEIL),
    );

    const ownErrors = perTrophy.get(i);
    let scale = 1;
    if (ownErrors !== undefined && ownErrors.length >= MIN_OWN_ERRORS) {
      let s = 0;
      for (const z of ownErrors) s += z * z;
      const rms = Math.sqrt(s / ownErrors.length);
      scale = clamp(1 + SCALE_SHRINK * (clamp(rms, 0.5, 2) - 1), SCALE_MIN, SCALE_MAX);
    }
    const sigma = clamp(Math.hypot(baseSigma * scale, stale ? STALE_WIDEN : 0), SIGMA_MIN, SIGMA_MAX);

    const q = shape.map((z) => round4(expit(clamp(l + z * sigma, -40, 40))));
    for (let j = 0; j < q.length; j++) q[j] = clamp(q[j] as number, 0, 100);
    for (let j = 1; j < q.length; j++) {
      if ((q[j] as number) < (q[j - 1] as number)) q[j] = q[j - 1] as number;
    }
    return { id, q };
  });
}

/** The persistence baseline's row: the fallback, digit for digit the kit's starter's arithmetic. */
function baselineRow(id: number, rate: number, horizonMinutes: number): { id: number; q: number[] } {
  const spread = BASELINE_SIGMA * Math.sqrt(Math.max(0, horizonMinutes) / 1440);
  const l = logit(rate);
  return { id, q: GAUSS_Z.map((z) => round4(expit(l + z * spread))) };
}

/** A case's output bytes: {trophies: [{id, q}, ...]} compact JSON and a newline, never another key. */
function outputBytes(c: TrophyCase): string {
  let rows: Array<{ id: number; q: number[] }>;
  try {
    rows = solve(c);
  } catch {
    const trophies = Array.isArray(c.trophies) ? c.trophies : [];
    const history = Array.isArray(c.history) ? c.history : [];
    const last = history.length > 0 ? (history[history.length - 1] as HistoryRead) : null;
    const rates = last !== null && Array.isArray(last.rates) ? (last.rates as Rates) : [];
    const horizon = finite(c.horizon_minutes) ? Math.max(0, c.horizon_minutes as number) : 0;
    rows = trophies.map((t, i) => {
      const id = finite(t.id) ? (t.id as number) : i + 1;
      const r = rates[i];
      return baselineRow(id, finite(r) ? r : RATE_FLOOR, horizon);
    });
  }
  return `${JSON.stringify({ trophies: rows })}\n`;
}

/** The moment of a case's last history read, or null. */
function lastReadTime(c: TrophyCase): number | null {
  const history = Array.isArray(c.history) ? c.history : [];
  for (let i = history.length - 1; i >= 0; i--) {
    const read = history[i] as HistoryRead;
    const t = timeOf(read.at) ?? timeOf(read.slot);
    if (t !== null) return t;
  }
  return null;
}

/** A diagnostic line on stderr, only when MAKEFUN_DEBUG=1: a graded run sets nothing and stays silent. */
function note(message: string): void {
  process.stderr.write(`main: ${message}\n`);
}

function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  for (const file of readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort()) {
    const id = file.slice(0, -'.in'.length);
    try {
      const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as TrophyCase;
      writeFileSync(join(OUTPUT_DIR, `${id}.out`), outputBytes(c));
    } catch (err) {
      process.stderr.write(`main: ${file}: ${err instanceof Error ? err.message : String(err)}\n`);
      process.exitCode = 1;
    }
  }
}

main();

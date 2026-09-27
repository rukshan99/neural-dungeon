# Floor 13 — The Watchtower

> *The dungeon is deployed. Somebody has to watch it, day and night, and know the difference between a bad day and a changed world.*

```
            ☀ the sky. Nothing above; the watch simply goes on.
            ┆  ◇ a lamp on the parapet: the Sequential Sentinel
   ┌────────┴────────┐     ┌──────────────────┐     ┌──────────────────────┐
   │ ☠ THE SILENT    │─────│ 13.4 THE ALARM   │─────│ 13.3 THE CANARY      │
   │     DRIFT       │     │      BELL        │     │        CAGE          │
   └─────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                               │
   ┌─────────────────┐     ┌──────────────────┐                │
   │ 13.1 THE LEDGER │─────│ 13.2 THE DRIFT   │────────────────┘
   │    OF SPANS     │     │      GLASS       │
   └────────┬────────┘     └──────────────────┘
            │  ▲ the service lift: up from the Engine Room (Floor 12),
            │  ┆ past twelve floors and the surface, to the tower
            ■  Floor 12, the bottom. The lift starts there.
```

At the bottom of the Engine Room, behind the last gauge, there is a lift you did not notice on the way down. It goes up. Past the Proving Grounds, past the Court and the Library, past the Threshold, past the surface, and stops in a round stone room at the top of a tower. From the parapet you can see the whole dungeon at once: every request coming in, every answer going out. The people who work here have never trained a model. They watch the one that is running.

Everything before this floor was about building a system that works. This floor is about the much longer period afterwards, when it is working and the world it works in keeps changing. Traffic shifts. A new kind of user arrives. A tool changes its output format. A model provider silently upgrades. Nothing crashes. Accuracy, if you could measure it, would be drifting down a fraction of a point a day, and you cannot measure it today, because the labels that would tell you arrive next week. Production ML is a monitoring problem with a model inside it, and the tools are the same ones you use for any service (traces, dashboards, alerts, canaries) plus one that is specific to models: watching the *distribution* of what comes in.

Everything on this floor runs offline against deterministic mocks and a simulated deployment in `assets/deployment.py`. The tracer wraps the same `LLM` protocol as Floors 9 to 11, so it works unchanged against a real adapter.

**You will learn:** traces and spans, an injectable clock, recording tokens and cost, redaction · distribution monitoring: PSI on quantile bins, the two-sample KS distance, Jensen-Shannon for categories, centroid drift for embeddings, and how to calibrate thresholds against sampling noise · canary releases: hashed assignment, unpaired bootstrap comparison, guardrails that fail closed, a promote/rollback/extend rule, ramps · alerting: nearest-rank percentiles, hysteresis, cooldowns, resolved events, error budgets and burn rate · the label-delay problem · the peeking problem and a sequential test that survives it.

**You need:** Python 3.11+ and numpy. No PyTorch. Floor 11 (bootstrap intervals, sample sizes) and Floor 12's room 4 (percentiles) are the habits this floor leans on.

---

## The lore of the Watchtower (read this before the rooms)

### A trace is a tree of spans

Every request the system serves should leave behind a record you can query afterwards. The unit is a **span**: a named, timed operation with a kind, a start and end, a bag of attributes, a status, and children. An agent run is a root span; each model call and each tool call is a child; a tool that itself calls a model has a grandchild. The tree is the **trace**.

```
agent.run  (chain)   371 ms   request_id=req-1 user=<redacted>
├── llm.complete (llm)   200 ms   model=scribe-v1 input_tokens=23 output_tokens=4 cost_usd=0.000129
├── lookup       (tool)   50 ms   arguments={"query": "watchtower"} status=ok
└── llm.complete (llm)   100 ms   ...
```

Three rules make a tracer trustworthy:

1. **Nesting comes from a stack.** `with tracer.span(...)` creates the span, attaches it to whatever span is on top of the stack (or makes it a root when the stack is empty), pushes it, and pops it on exit. No ids to thread through your code.
2. **The clock is injected.** `Tracer(clock)` takes any callable returning seconds. Production passes `time.perf_counter`; tests pass a fake they advance by hand, and every duration assertion becomes exact. A tracer that reads the wall clock itself cannot be tested.
3. **Errors are recorded and re-raised.** The span gets `status = "error"` and `error.type` / `error.message` attributes; the span is still closed and popped (`finally`); the exception propagates. Swallow it and every failure becomes a quiet success. Forget the pop and every later span becomes a child of the corpse.

Model calls are the expensive spans, so they record what the completion reports: `usage.input_tokens`, `usage.output_tokens`, and a cost at a per-million-token price. A summary of a trace is one walk of the tree: total time, time in models, time in tools, tokens, cost, call counts, errors, depth. Two more habits: serialize with a **fixed field order** (the same trace produces the same bytes, so log queries and diffs do not break when a field moves), and **redact at the tracer**, by attribute name, before the trace leaves the process. Traces get shipped to dashboards; the user's email address should not.

### Watching the inputs: distribution monitoring

Labels arrive late or never. Inputs arrive now. So the first thing a tower measures is whether the distribution of today's inputs still looks like the distribution the model was validated on. Every instrument below compares a **current** sample with a fixed **reference** sample (the validation set, or launch week). Fix the reference; re-baseline deliberately after a promotion, never automatically.

**Population Stability Index** (one numeric feature). Cut the reference into `n` quantile bins so each holds about 1/n of it, measure the fraction of the current sample in each bin, and sum

```
PSI = Σ_i (c_i − r_i) · ln(c_i / r_i)          fractions clipped to ≥ eps first
```

Zero when the fractions match, growing with the shift. The rule of thumb from credit scoring still serves: **< 0.1 stable, 0.1–0.25 moderate, ≥ 0.25 major**. Always bin with the reference's edges, computed once. If you re-bin on the current sample, the yardstick moves with the thing you are measuring.

**Kolmogorov-Smirnov distance** (numeric, no bins). The largest vertical gap between the two empirical CDFs, `max_x |F_ref(x) − F_cur(x)|`, in [0, 1]: 0 for identical samples, 1 for samples that do not overlap. Evaluate both CDFs on the pooled sorted values; `np.searchsorted(sorted_a, grid, side="right") / len(a)` is F_a on the grid.

**Jensen-Shannon divergence** (categorical: tool-call names, refusal vs answer, intent labels). With `m = (p + q) / 2`,

```
JS(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m)         KL(p ‖ m) = Σ p_i ln(p_i / m_i)
```

Unlike KL it is symmetric and bounded: 0 for identical distributions, ln 2 for disjoint ones. A category the reference never contained is drift in itself; do not drop it silently.

**Embedding drift** (vectors). Compare the two clouds' centroids by cosine, and their mean norms: `(1 − cos(c_ref, c_cur)) + |Δ mean norm| / mean norm`. Rotation shows up in the first term, scaling in the second; a new embedding model version shows up as a step in both.

**Calibrate against noise, not against hope.** Two samples from the *same* distribution do not give PSI 0. With 10 bins, n = 500 current against n = 5000 reference, PSI's null distribution has mean ≈ 0.02 and a tail that essentially never passes 0.1 (it behaves like `(1/n_cur + 1/n_ref) · χ²₉`). KS is noisier: its 95% null quantile at those sizes is ≈ 0.06 and a threshold of 0.10 will false-alarm about once in 5000 feature-days, which across four features, twenty days and several deployments is not rare. This floor's monitor defaults to PSI 0.25 / KS 0.15; the boss lowers PSI to 0.1 (a canary is cheap) and keeps KS at 0.15. Check per feature and **name the feature that moved**; a monitor that says "something drifted" has told you to go look, one that says "feature 2" has told you where.

### Deciding with statistics: canary releases

A **canary** sends a sliver of live traffic to a candidate and compares it with the incumbent on the same days. Four small pieces:

**Assignment.** `sha256(f"{salt}:{request_id}")` → first 8 bytes as an integer → divide by 2⁶⁴ → a uniform `u` in [0, 1); canary if `u < fraction`. Deterministic (the same request or user always lands in the same arm, with no lookup table), stateless, and **monotone under ramps**: comparing one `u` against every fraction means everyone in the 5% canary is still in the 25% canary. New experiment, new salt.

**Comparison.** The two arms saw *different* requests, so Floor 11's paired bootstrap does not apply. Resample each arm with replacement on its own, compute `mean(canary*) − mean(baseline*)` per round, and take percentiles for the interval; the two-sided p-value is `min(1, 2·min(P(δ* ≤ 0), P(δ* ≥ 0)))`. Feel the sizes: a 100-request canary against 900 baseline requests at 85% accuracy has an interval about ±7 points wide. It cannot see a two-point change, and it is not meant to. It catches disasters cheaply; you ramp for the rest.

**Guardrails.** Quality is not the only thing that breaks. p95 latency, error rate and cost per request each get a hard limit (the SLO). A metric the canary did not report is a **failed** guardrail: no gauge, no promotion. Guardrails need no labels, so they are checked the day the traffic happens, pooled over the canary's requests so far rather than judged on one day of fifty.

**The rule**, written once, as code, so that nobody argues about it at 2 a.m.:

```
rollback   if any guardrail fails, or the CI is entirely below 0
promote    if the CI is entirely above 0, delta ≥ min_effect, and guardrails hold
extend     otherwise: not enough evidence yet, keep collecting
```

`min_effect` separates "statistically significant" from "worth shipping".

**The ramp.** `1% → 5% → 25% → 50% → 100%`. Each "promote" moves one step up; "extend" stays; "rollback" goes to 0. One rule the boss enforces: after a step, **wait for fresh evidence**. Tomorrow is the first day served at the new fraction and its labels arrive `label_delay` days later; deciding again before then is deciding twice on the same numbers.

### Alerting that people will not mute

Alerts are the part of monitoring that touches humans, so their failure modes are human ones: the bell that rings six times for one incident gets muted, and the muted bell misses the real one.

- **Percentiles, nearest rank** (as on Floor 12): sort, take the element at 1-based rank `ceil(p/100 · n)`. p50 is the typical request; p99 is the one in the contract. A tail that grows on 3% of requests moves p99 and leaves p50 exactly where it was.
- **Hysteresis**: fire when the value crosses a high line, clear only when it drops below a *lower* line, do nothing in between. A metric hovering at 0.19, 0.21, 0.19, 0.22 around a fire line of 0.20 with a clear line of 0.15 rings once and resolves once. A plain threshold flaps.
- **Cooldown**: a second fire for the same rule within N minutes of the first is suppressed (counted, not paged), and so is its quiet resolution.
- **Resolved events**: say when it is better. An incident that never resolves is a person still awake.
- **Error budgets**: an SLO of 99.9% over 100,000 requests *allows* 100 failures. `remaining = (allowed − failures) / allowed`; `burn rate = observed failure rate / allowed failure rate`. Burn 1.0 spends the budget exactly at the end of the window; 2.5 spends it in 40% of the window. Alert on burn rate over two windows (fast burn pages, slow burn tickets), not on individual failures.

### The label-delay problem

Accuracy on day *t* is knowable on day *t + d*, where *d* is however long labels take: a human review queue, a customer complaint, a downstream outcome. Two consequences. First, **date quality by the day of the request**, not the day the label arrived, and draw the last *d* days as pending, not as zero. Second, under slow drift the order of events is fixed: the inputs move first, quality suffers later (the drift has to grow before the model is wrong in a place it is actually asked), and quality is *seen* *d* days after that. A tower that only watches accuracy is always `d + (time for the drift to hurt)` days behind, which for the boss's simulation is about three weeks. That is the boss's weakness and the floor's thesis: watch the inputs.

### The peeking problem (the secret room)

A 95% interval is a promise about one look. Check the canary every morning for a month and stop the first time the interval excludes zero, and you have taken thirty looks; the chance at least one of them lies is closer to 20% than 5%. The boss peeks daily; before it learned to wait for a minimum canary sample, one calibration seed in twenty rolled back a better model on a chance interval of (−0.061, −0.001). Fixed-horizon tests fix this by never peeking. **Sequential tests** are designed for peeking: Wald's SPRT adds a log-likelihood-ratio term per observation and stops at bounds `ln(β/(1−α))` and `ln((1−β)/α)` that hold the error rates no matter when you look, usually well before the fixed-horizon sample size. The trick for two arms is to pair the outcomes and keep only the **discordant** pairs, where exactly one arm succeeded: under "no difference" the winner of such a pair is a fair coin.

---

## Rooms

Run `dungeon enter 13` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code, then run that room's trial.

### 13.1 The Ledger of Spans — `rooms/room_1_ledger_of_spans.py`

The lift opens onto a room of ledgers. Build `Tracer.span` (the stack, the injected clock, the error path), `record_llm_call` (tokens and cost from `completion.usage`), `record_tool_call`, `summarize`, `to_json`/`from_json` with `FIELD_ORDER`, and `redact_attributes`. The trial hands you a clock it advances by hand, so every duration it checks is exact: a 371 ms agent run with 300 ms in models and 50 ms in a tool.

```
dungeon trial 13 room_1
```

### 13.2 The Drift Glass — `rooms/room_2_the_drift_glass.py`

Smoked glass on the north wall. `histogram_bins` and `bin_fractions`, then `psi` and `psi_verdict`, `ks_statistic`, `js_divergence` and `category_fractions`, `embedding_drift`, and the `DriftMonitor` that ties them together per feature. The trial checks the hand cases (a KS of exactly 0.5, a JS of exactly ln 2), the noise floor (ten unshifted batches, three seeds, not one false alarm at the defaults) and that a one-standard-deviation shift on feature 1 is flagged *as feature 1*.

```
dungeon trial 13 room_2
```

### 13.3 The Canary Cage — `rooms/room_3_the_canary_cage.py`

A brass cage with a slot in the top. `assign_arm` by sha256 (deterministic, fraction respected over 20,000 ids, monotone under ramps), `compare_arms` with an unpaired bootstrap (a 100-request canary must be honest about how little it can see), `guardrail_check` that fails closed, `release_decision` with every branch exercised, and `ramp_schedule`/`advance_ramp`.

```
dungeon trial 13 room_3
```

### 13.4 The Alarm Bell — `rooms/room_4_the_alarm_bell.py`

A bronze bell and a ledger of crossed-out entries. `RollingWindow` with nearest-rank percentiles, `Threshold` with hysteresis (a hovering metric rings once), `AlertManager` with a cooldown and resolved events, and `error_budget`: 99.9% over 100,000 requests is exactly 100 allowed failures.

```
dungeon trial 13 room_4
```

---

## Boss: The Silent Drift

```
                                                      "No day is the day it changed.
      .       .        .       .        .       .      Ask your accuracy. It will tell
         .        .        .       .        .          you nothing happened, and it will
      .     .   .    .    .   .    .    .    .  .      go on saying so for weeks, and it
     . . . . . . . . . . . . . . . . . . . . . . .     will be telling the truth about
    ...............................................    the only days it can see."
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   The Silent Drift moves the world
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  under a model a little each day.
```

**Weakness:** watching the inputs, not just the accuracy, and deciding with statistics.

The simulation (`assets/deployment.py`): sixty days of 500 requests, four features each. The true label is a noisy rule with a hinge on `x2`; the deployed scribe ignores `x2`, which cost it about a point while `x2` stayed small. From day 20 the mean of `x2` climbs by 0.06 a day (about an eighth of its spread), so the deployed scribe's accuracy slides from 91% to about 85% over the following two weeks and to 81% a few days after that: under half a point a day, inside daily noise of about 1.3 points. Nothing else changes. Labels arrive five days late. A retrained scribe that knows the hinge is waiting, and so are a hasty one that is 25 points worse, a slow one, and a brittle one.

You build `Watchtower.observe_day(day, features, labels_delayed)`, which does five things in order: the glass, serving with arm assignment, guardrails on today's canary traffic, scoring the day whose labels just arrived, and the release decision.

- **Phase 1:** the glass sees the drift within 7 days of onset on three seeds and never before it; forty days without drift log nothing. (The reference thresholds, PSI 0.1 / KS 0.15, detect on day 1–3 of the drift with zero early alarms across thirty seeds.)
- **Phase 2:** the retrained scribe is promoted, through a 10% → 50% → 100% ramp, with every promote carrying a CI that excludes zero and guardrails that passed, no quality decision before `MIN_CANARY_SCORES` labelled canary requests exist, and no second step until fresh labels exist. The hasty scribe is rolled back on the numbers the first day the sample allows it. The slow and brittle scribes are rolled back on guardrails *before any label arrives*.
- **Phase 3:** `decision_log` entries with `day, signal, value, decision, reason` (and the evidence in `details`); `render_decision_log` turns it into a markdown table. The bell rings about the deployed scribe's error rate roughly three weeks after the glass spoke.
- **Phase 4:** `DRIFT_PROPHECY`: which signal moves first, which percentile sees a 3% tail, and whether a 0.5% canary at 1000 requests a day can see a two-point change in a week. Do Floor 11's arithmetic before you answer the third one.

```
dungeon fight 13
```

## Secret room: The Sequential Sentinel *(optional)*

A lamp on the parapet. Implement Wald's SPRT on discordant pairs: `discordant_probability`, `wald_bounds`, a `SequentialSentinel` with `update`/`feed`, and `fixed_horizon_pairs` for comparison. The trial runs 200 A/A streams (the sentinel declares a winner about 3% of the time at α = 0.05; a naive daily z-test peeked at for a month does so about 20% of the time) and 200 streams with a real five-point lift (right about 80% of the time, stopping after about 500 pairs against a fixed-horizon 711).

```
dungeon trial 13 --secret
```

## Loot

Clear the four rooms and defeat the Silent Drift to unlock:

- **Monitoring Checklist** — `loot/monitoring_checklist.md`. What to log per request, what to watch at which percentile, drift signals and thresholds with their noise floors, the canary/ramp/rollback recipe, alert hygiene, the label-delay problem, and an incident checklist.
- **Watchtower Kit** — `loot/watchtower_kit.py`. A dependency-free, importable copy of the tower's tools: `Tracer`, PSI/KS/JS/embedding drift and `DriftMonitor`, `assign_arm`, `compare_arms`, `guardrail_check`, `release_decision`, hysteresis alerts, `error_budget`, and the `SequentialSentinel`.

## Stuck?

- `dungeon hint 13 boss` reveals one hint at a time (three per room).
- The trial failure messages quote the day, the value and the rule they expected. Read the decision log your tower wrote; that is what it is for.
- `solutions/` exists. Use it the way you would use a walkthrough: after an honest attempt, and to compare, not to copy.

When `dungeon map` shows the Watchtower cleared, there are no more stairs in either direction. There is only tomorrow's traffic, and you know how to watch it.

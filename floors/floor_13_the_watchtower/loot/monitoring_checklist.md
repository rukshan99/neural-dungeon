# Monitoring Checklist

*Loot from Floor 13. Print it, pin it above the pager.*

## Log this per request (the span attributes)

| attribute | why |
|---|---|
| `request_id`, `trace_id`, parent span | join everything else to it later, including labels that arrive next week |
| `model` (name AND version), prompt/template version | "the model got worse" is meaningless without a version |
| `input_tokens`, `output_tokens`, `cost_usd` | cost drifts too; a retrieval bug that doubles context is a cost incident first |
| latency per span: total, time-to-first-token, each tool | the tail lives in one span; find which |
| `stop_reason` / finish reason, `status`, `error.type` | truncation and refusals are quality regressions that need no labels |
| tool names called, tool statuses | a changed tool-call mix is drift you can see the same day |
| the arm (`baseline` / `canary`) and canary fraction | or you cannot compare arms afterwards |
| a hash of the input, and the embedding if you compute one | drift on the inputs; never the raw text in the trace store |
| user/tenant id **hashed**, never emails or keys | redact at the tracer, not at the dashboard |

Field order in the serialized span is fixed. Redaction is a function of the attribute name and runs before the trace leaves the process.

## Watch this, at this percentile

| signal | statistic | why not the mean |
|---|---|---|
| latency, TTFT | p50 for the typical user, **p95/p99 for the SLO** | one slow request in a hundred is invisible to the mean and to the median |
| error / timeout / refusal rate | rate over a rolling window | counts, not "did it happen today" |
| tokens and cost per request | p50 and p99, plus the daily sum | the daily sum pays the bill; the p99 finds the runaway prompt |
| tool-call mix, stop reasons | fractions per category, JS against reference | categorical drift |
| input features / embeddings | PSI, KS, centroid cosine against a fixed reference | covariate shift, visible today |
| quality (accuracy, judge score) | mean with a CI, **dated by the day the request happened**, not the day the label arrived | the label delay |

Nearest-rank percentile: sort, take rank `ceil(p/100 * n)`. It never interpolates, so it is always a latency that happened.

## Drift signals and thresholds

- **Reference sample**: the validation or launch-week distribution. Fix it. Re-baseline deliberately (after a promotion), never automatically.
- **PSI** on quantile bins of the reference: `sum (c - r) ln(c / r)`, fractions clipped to eps. Rule of thumb: < 0.1 stable, 0.1–0.25 moderate (start a canary, look), ≥ 0.25 major (page). Sampling noise at n = 500 vs 5000 with 10 bins is about 0.02 ± 0.02; 0.1 is safely above it.
- **KS**: `max |F_ref − F_cur|`. Noise scales like `1.36 / sqrt(n_eff)`; at n = 500 vs 5000 the 95% null quantile is ≈ 0.06, so 0.1 false-alarms across many features and days; 0.15 does not.
- **JS** for categories (nats, max ln 2): 0.05 is a visible mix change; new category never seen in the reference = alert on its own.
- **Embedding drift** = `(1 − cos(centroids)) + |Δ mean norm| / mean norm`. Rotation vs scaling; a new embedding model version shows up here as a step.
- Check per feature; **name the feature** that moved. One monitor per model input, one per output category.
- Pool a window of batches when a single batch is small; accept the extra day of lag.

## Canary / ramp / rollback recipe

1. Assign by `sha256(salt:request_id)` → uniform in [0, 1) → canary if below the fraction. Deterministic, stateless, monotone under ramps. New experiment, new salt.
2. Ramp `1% → 5% → 25% → 50% → 100%`. Each step waits for **fresh** evidence: at least `label_delay + 1` days after the step, so labels from traffic at the new fraction exist.
3. **Guardrails first, every day, no labels needed**: p95 latency, error rate, cost per request, each with a hard limit. Pool over the canary's requests so far (a rolling window) rather than judging one day of 50 requests. A metric you did not measure is a failed guardrail.
4. **Quality when labels arrive**: unpaired bootstrap of `mean(canary) − mean(baseline)` (the arms saw different requests). Report delta, CI, p, both n.
5. **The rule, as code**: rollback if a guardrail fails or the CI is entirely below 0; promote if the CI is entirely above 0, `delta ≥ min_effect` and guardrails hold; else extend.
6. Rollback is a state, not an event: the candidate serves nothing, a human is told, the log says why.
7. Sample-size sanity before you start (Floor 11): a 0.5% canary at 1000 requests/day is 35 requests a week; its standard error near 90% accuracy is 5 points. It catches disasters, not two-point wins. Ramp for the wins.
8. If you look every day, use a sequential test (the secret room) or accept ~20% false alarms over a month of peeking.

## Alert hygiene

- **Hysteresis**: fire above one line, clear below a lower one. A metric hovering at the line pages once.
- **Cooldown**: no second page for the same rule inside N minutes. Count suppressions; they are data.
- **Resolved events**: say when it is better. Unresolved alerts are people still awake.
- **Burn rate, not failures**: 99.9% over 100k requests allows 100 failures; `burn = observed failure rate / allowed rate`. Page at a fast burn over a short window (e.g. 14× over 1 h), ticket at a slow burn over a long one (e.g. 1× over 3 days).
- Every alert names the runbook, the dashboard and the value it saw. An alert with no number is an opinion.
- Review the alert ledger monthly: delete rules that only ever flapped or were never acted on.

## The label-delay problem

- Accuracy on day *t* is knowable on day *t + d*. Date quality metrics by the day of the **request**, and draw the last *d* days as "pending", not as zero.
- Under slow drift the inputs move first, quality later, and quality is seen *d* days after that. A tower that only watches accuracy is `d + (time for drift to hurt)` days behind.
- Log predictions with request ids so labels can be joined later; keep the arm assignment in the same record.
- Guardrails (latency, errors, cost, refusals, truncation) need no labels: they are your same-day signals. Proxy quality (judge scores, user edits, retries, thumbs) arrives faster than gold labels and drifts less than you fear.
- When labels are sampled (human review), sample **uniformly at random**, stratified by arm, and record the sampling rate.

## Incident checklist

1. **Confirm it is real**: which signal fired, what value, which percentile, which window? Compare with the same hour last week.
2. **What changed?** Model version, prompt version, retrieval index, tool, dependency, traffic mix, a canary step. The decision log tells you; if it does not, that is the first fix after the incident.
3. **Stop the bleeding**: roll back the last change (canary fraction to 0). Rollback before diagnosis.
4. **Scope**: which tenants, arms, tools, feature ranges? Slice the traces by the attribute that moved.
5. **Communicate**: one status line with a number and a next update time.
6. **Fix forward** only behind a new canary.
7. **Write it down**: timeline from the decision log, the signal that caught it, the signal that *should* have caught it earlier, and the threshold or alert you are changing. Re-baseline the reference sample only if the new world is the intended one.

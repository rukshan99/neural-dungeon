# Evaluation Cheat Sheet

*Loot from Floor 11. One page. Read it before you trust a number.*

## Metrics: the formula and when it lies

Confusion matrix: `cm[i, j]` = true class i, predicted class j. Binary: TP = cm[1,1], FP = cm[0,1], FN = cm[1,0], TN = cm[0,0].

| metric | formula | lies when |
|---|---|---|
| accuracy | (TP + TN) / N | classes are imbalanced: the majority guesser scores the majority rate |
| precision | TP / (TP + FP) | you predict almost nothing positive (few flags, all correct, most positives missed) |
| recall | TP / (TP + FN) | you predict everything positive (recall 1.0, precision at the base rate) |
| F1 | 2PR / (P + R) | the negative class matters as much as the positive one (F1 ignores TN) |
| macro-F1 | mean over classes of one-vs-rest F1 | a tiny class dominates the mean because it is one of K equal terms; that is also the point |
| balanced accuracy | mean over classes of recall | never about ranking; a majority guesser scores exactly 1/K |
| perplexity | exp(mean NLL) | compared across different tokenizers; or computed as mean(exp(nll)) |
| ECE | Σ (n_b/N) · \|acc_b − conf_b\| | few bins on little data (noisy); tells you nothing about accuracy |
| win rate (judge) | wins / comparisons | the judge has a position or length bias; the opponent is weak |

Zero-division conventions: precision with no predicted positives = 0.0, recall with no positives = 0.0, F1 when P + R = 0 is 0.0. Pick conventions once, document them, never emit NaN.

Perplexity: `exp(mean(nll))`, average FIRST. Uniform over V tokens = V. Perfect = 1.

## Error bars: the rule of thumb

- Standard error of a proportion: `SE = sqrt(p(1−p)/n)`. 95% interval ≈ `p ± 1.96·SE`.
- Difference of two systems: `SE_diff = sqrt(SE_a² + SE_b²)` (unpaired) or `sd(b − a)/sqrt(n)` (paired: much smaller when the systems agree on most items).
- A gap needs ~2 standard errors to be distinguishable from zero.

| n | 95% half-width at p = 0.8 | can distinguish 80% vs 84%? |
|---|---|---|
| 50 | ± 0.11 | no |
| 200 | ± 0.055 | no |
| 1000 | ± 0.025 | barely |
| 2000 | ± 0.018 | yes |

- Sample size for margin m: `n = z² p(1−p) / m²`. At p = 0.8: m = 0.05 needs 246; m = 0.02 needs 1537.
- Bootstrap (any statistic): resample n items with replacement, recompute, repeat 2000×, take the 2.5th and 97.5th percentiles.
- Paired bootstrap: resample item INDICES once per round and apply them to both systems. Significant when the interval excludes 0.
- Fix n before looking. Do not stop when the interval first excludes zero.

## LLM-as-judge design checklist

- [ ] Rubric with named criteria and an integer scale; both ends of the scale in the prompt.
- [ ] Ask for JSON `{"score": int, "rationale": str}`; the rationale is for debugging disagreements.
- [ ] Parse robustly: strip fences, find the first balanced `{...}`, regex fallback, raise only when there is truly no score. A parser that silently returns 0 corrupts the leaderboard.
- [ ] Range-check the score against the scale.
- [ ] Pairwise: judge (a, b) AND (b, a). Consistent winner or "tie". Never trust one order.
- [ ] Length: truncate both candidates to the shorter one's word count, or instruct the judge to penalise filler; verify with a swap either way. Report a filler ratio next to the win rate.
- [ ] Self-preference: do not judge a model with itself when you can avoid it; if you must, say so.
- [ ] Validate the judge on a human-labelled set: report Cohen's kappa `(p_o − p_e)/(1 − p_e)` AND Spearman (average ranks for ties, then Pearson). Kappa ≥ 0.6 is decent agreement; a harsh-but-consistent judge has low kappa and high Spearman.
- [ ] Pin the judge model version. A judge upgrade is a metric change.
- [ ] Sample and read the judge's rationales every release. Judges drift and prompts rot.

## Eval hygiene

- **Golden set**: curated, versioned, tagged, run on every change. You will look at its failures constantly, so it WILL leak into your decisions. That is fine; that is what it is for.
- **Held-out set**: never used to pick a prompt, a model or a hyperparameter. Score it rarely. The gap `golden − heldout` is your contamination alarm (this floor flags > 0.3).
- **Contamination**: memorised golden cases, benchmark text in training data, test questions pasted into the system prompt. Check with fresh items, paraphrases, or a held-out gap.
- **Baseline**: save every report as JSON with the case-set version and the model/prompt version. A regression is a case that passed before and fails now, or an overall drop beyond the noise (know your noise: bootstrap it).
- **Per-tag scores** localise regressions; an overall score only tells you that something moved.
- **Keep the outputs**, not just the scores. A report you cannot debug is a rumour.
- **Never crash**: a system that raises on one case scores 0 on that case; the rest of the run still happens.
- **Stratify** splits by tag so every capability appears on both sides.

## Goodhart failure catalogue

*When a measure becomes a target, it ceases to be a good measure.* The metric does not change; the set of systems that score well on it grows to include ones that do not do the job.

| target | how it gets gamed | the second number |
|---|---|---|
| accuracy | predict the majority class | balanced accuracy, per-class recall |
| recall | flag everything | precision, F1 |
| judge win rate | longer answers, confident tone, restating the question | length-controlled judging, filler ratio, human spot checks |
| golden-set score | memorise the cases; overfit the prompt to them | held-out score, the gap |
| benchmark perplexity | train on the benchmark | contamination checks, n-gram overlap with training data |
| mean rating | solicit only happy users; suppress the rating prompt after failures | response rate, rating distribution, unsolicited feedback |
| latency p50 | time out or drop the hard requests | p99, error and timeout rates |
| "resolved" tickets | close without fixing | reopen rate, follow-up contacts |
| test pass rate | delete or skip failing tests | test count over time, coverage |
| tokens per second | shorter outputs, smaller vocab | quality metrics at fixed length |

Rule: never ship on one number. Choose the second number so that the obvious way to game the first one makes the second one worse.

## Calibration (bonus)

- Confidence = max softmax probability. Correct = argmax == label.
- `ECE = Σ_b (n_b / N) · |accuracy_b − mean_confidence_b|` over confidence bins `(b/n, (b+1)/n]`.
- Temperature scaling: `softmax(logits / T)`, choose T minimising NLL on held-out data (a grid is fine). T > 1 softens; argmax never changes; accuracy untouched.
- Report ECE next to accuracy whenever a downstream system acts on the confidence (routing, abstention, ranking).

# Floor 11 — The Proving Grounds

> *Every champion below has a number. Almost none of the numbers mean what they say.*

```
            ▼ stairs from Floor 10
            │
   ┌────────┴────────┐     ┌──────────────────┐     ┌──────────────────┐
   │ 11.1 THE SCALES │─────│ 11.2 THE BOOT-   │─────│ 11.3 THE JUDGE'S │
   │                 │     │   STRAP ORACLE   │     │      BENCH       │
   └─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                             │
   ┌─────────────────┐     ┌──────────────────┐              │
   │ ☠ THE GORGON'S  │─────│ 11.4 THE GOLDEN  │──────────────┘
   │     ARENA       │     │      TRIALS      │
   └────────┬────────┘     └──────────────────┘
            ┆  ◇ a glint behind a loose stone: the Calibration Mirror
            ▼  stairs down to Floor 12
```

The stairs open onto sand. This floor is an arena, and everything that fought its way down from the floors above is here to be measured: classifiers, language models, retrieval pipelines, agents. A steward with a brass balance weighs each one and calls out a number. The crowd cheers for the big numbers. Nobody in the crowd asks what the number is *of*.

You have built things on ten floors. From here down, the question is no longer "can I build it?" but "is it any good, and how would I know?" That question is most of the job. A model change that "feels better" is a coin flip until you measure it, and a measurement is a coin flip until you know its error bars. Then the hard part: the moment a number decides what ships, everyone and everything starts optimising the number instead of the thing, and the number quietly stops meaning anything. There is a Gorgon down here who does exactly that with her eyes.

Everything on this floor runs offline against deterministic mock models from `dungeon/artifacts/llm.py`. The judges you build here work unchanged against a real model: swap the mock for the adapter in `dungeon/artifacts/providers.py` and the code does not change.

**You will learn:** confusion matrices, precision, recall, F1, macro and balanced metrics · why accuracy lies on imbalanced data · perplexity · sampling error and bootstrap confidence intervals · paired comparisons and sample sizes · LLM-as-judge: rubrics, parsing, position bias, swap debiasing, agreement with humans (Cohen's kappa, Spearman) · eval harness design: golden sets, held-out sets, regression baselines · Goodhart's law and how metrics get gamed · calibration and temperature scaling.

**You need:** Python 3.11+ and numpy. No PyTorch on this floor.

---

## The lore of measurement (read this before the rooms)

### Everything starts with counts

A classifier's predictions against the truth fit in one table, the **confusion matrix**. Rows are the true class, columns are the predicted class:

```
cm[i, j] = number of examples whose TRUE class is i and PREDICTED class is j
```

For a binary task with class 1 as "positive":

```
                 predicted 0     predicted 1
    true 0          TN              FP          (false alarm)
    true 1          FN              TP          (missed one)
```

Every metric in room 1 is a ratio of these four counts:

| metric | formula | reads as |
|---|---|---|
| accuracy | (TP + TN) / N | fraction right |
| precision | TP / (TP + FP) | of what I flagged, how much was real |
| recall | TP / (TP + FN) | of what was real, how much I flagged |
| F1 | 2·P·R / (P + R) = 2TP / (2TP + FP + FN) | harmonic mean; low if either P or R is low |
| macro-F1 | mean over classes of the one-vs-rest F1 | every class counts equally |
| balanced accuracy | mean over classes of the per-class recall | every class counts equally |

Denominators can be zero. The conventions this floor uses (and the trial checks): precision with no predicted positives is 0.0, recall with no actual positives is 0.0, F1 when P + R = 0 is 0.0. Never NaN, never an exception. Decide these conventions once and write them down; a metric that silently produces NaN on an edge case will one day average into a leaderboard.

### Why accuracy lies

Take 1000 examples, 950 negative and 50 positive. A "model" that always predicts negative has:

```
accuracy          = 950 / 1000 = 0.95
recall (class 1)  = 0 / 50     = 0.00
balanced accuracy = (1.00 + 0.00) / 2 = 0.50
macro-F1          = (F1_class0 + F1_class1) / 2 = (0.974 + 0) / 2 = 0.487
```

Ninety-five percent accurate, and it has learned nothing. Accuracy weighs every *example* equally, so on imbalanced data it is mostly a measurement of the majority class. Balanced accuracy and macro-F1 weigh every *class* equally, which is why a majority guesser scores exactly 1/K on balanced accuracy. Whenever the classes are unbalanced (fraud, defects, rare diseases, the one intent that matters), report accuracy next to a balanced metric, and read the gap.

### Perplexity: the language model's metric

A language model is trained to minimise the mean **negative log-likelihood** (NLL) of the correct next token:

```
NLL = -(1/T) · Σ_t log p(x_t | x_<t)
perplexity = exp(NLL)
```

Perplexity is the effective number of equally likely choices the model is hedging between per token. A uniform model over a vocabulary of V tokens assigns log p = −log V everywhere, so its perplexity is exactly V: "as confused as a V-sided die". A perfect model has perplexity 1.

Two traps. First, the order of operations: **average the log-losses, then exponentiate once.** `exp(mean(nll))` is not `mean(exp(nll))`; for losses `[0, ln 100]` the first is 10 and the second is 50.5. Second, perplexity depends on the tokenizer: a model with a bigger vocabulary sees fewer, harder tokens per sentence, so perplexities are comparable only when the tokenization is identical. Compare per-token NLL across tokenizers only after normalising to bits per character or per byte.

### Every score is an estimate

You ran the system on n items and got 80%. Run it on a *different* n items and you will get a different number. How different? For a proportion p estimated from n independent items the **standard error** is

```
SE = sqrt( p · (1 − p) / n )
```

At n = 50 and p = 0.8 that is 0.057, so the 95% interval (roughly ±1.96·SE) is 80% ± 11 points. Comparing two systems, the error of the *difference* adds the variances: SE_diff = sqrt(SE_a² + SE_b²) ≈ 0.077 at n = 50. A four-point gap is half a standard error. **At n = 50 you cannot tell 80% from 84%.** At n = 2000 the same gap is 3.3 standard errors and the difference is real.

How many items do you need to pin a proportion near p down to ± m? Invert the formula:

```
n = z² · p · (1 − p) / m²          z = 1.96 for 95%
```

For p = 0.8 and m = 0.02 that is 1537 items. Halving the margin quadruples the sample.

### The bootstrap: resample fate

The formula above assumes a proportion. For a median, an F1 score, a judge's mean rating or anything else, the **bootstrap** gives you the sampling error with no formula at all:

1. Treat your sample of n scores as the population.
2. Draw n scores from it *with replacement*. Compute the statistic.
3. Repeat B times (B = 2000 is plenty).
4. The 2.5th and 97.5th percentiles of the B recomputations are a 95% **percentile confidence interval**.

```python
idx = rng.integers(0, n, size=(n_boot, n))          # every resample's indices at once
stats = np.array([statistic(values[row]) for row in idx])
low, high = np.percentile(stats, [2.5, 97.5])
```

Comparing two systems A and B on the *same* items, do a **paired bootstrap**: resample item *indices* and take both systems' scores at those indices together, computing `mean(b[idx] − a[idx])` each round. Items both systems got right or both got wrong cancel out, so the interval reflects only the items where they *differ*. A small but perfectly consistent gain (B is better on every single item by 0.05) gives an interval of exactly (0.05, 0.05); resampled independently it would look like noise. If the interval excludes zero, the difference is significant at that level. The two-sided **p-value** in the percentile flavour is twice the smaller of P(delta* ≤ 0) and P(delta* ≥ 0) over the bootstrap deltas, capped at 1.

Whatever you do, decide the sample size *before* looking at the results, and never stop collecting when the interval first excludes zero.

### LLM-as-judge

When there is no exact answer to compare against (a summary, an explanation, a code review), you ask a model to grade. Three engineering problems, one robe:

**The prompt.** A **rubric** is a list of criteria plus a scale. Put the criteria and both ends of the scale in the system message. Ask for the *rationale* as well as the score; it makes disagreements debuggable. Ask for a machine-readable format: `{"score": 4, "rationale": "..."}`.

**The parse.** Models decorate JSON: fences, "Certainly! Here is my judgement:", a trailing sentence. Find the first balanced `{...}` that parses, fall back to a regex for `"score": 4`, and raise only when there is truly no score. A parser that returns 0 on a formatting quirk will silently tank a system's score and you will spend a day looking for the bug in the wrong place.

**The bias.** Judges have well-documented habits. **Position bias**: they prefer the first candidate when the contest is close. **Length bias**: they prefer the longer answer. **Self-preference**: they prefer text in their own style. Position bias has a clean fix: judge (a, b) *and* (b, a). If both verdicts name the same text, trust them. If the judge contradicts itself, the honest verdict is a **tie**. Two calls instead of one; worth it. (Length bias needs a different fix; the boss has it.)

**Measuring the judge.** A judge is a system; evaluate it against humans on a labelled set. Raw agreement is misleading because two raters using a 5-point scale will agree some of the time by luck. **Cohen's kappa** corrects for that:

```
kappa = (p_o − p_e) / (1 − p_e)
p_o = fraction of items where the two raters agree
p_e = Σ_k  P(rater A says k) · P(rater B says k)        expected agreement by chance
```

1.0 is perfect agreement, 0.0 is chance level, negative is systematic disagreement. For ordinal scores also report **Spearman's rank correlation**: convert each sequence to ranks, giving tied values the *average* of the ranks they span (`[10, 20, 20, 30]` → `[1, 2.5, 2.5, 4]`), then compute Pearson's correlation on the ranks. A judge that is consistently one point harsher than humans has a low kappa but a Spearman near 1; both numbers tell you something.

### Eval harness design

An eval harness is a loop with excellent bookkeeping:

```python
for case in cases:                       # EvalCase(id, input, expected, tags)
    output = system_fn(case.input)
    score  = scorer(case, output)        # 0.0 .. 1.0
    remember everything
```

The bookkeeping is the product:

- **Cases carry tags.** A per-tag score tells you *which* capability regressed.
- **Scorers are pluggable**: exact match, contains, numeric tolerance, an LLM judge. Pick per case set.
- **Keep the evidence.** Store the actual output next to the expected one; a report with only a score cannot be debugged.
- **Never crash.** A system that raises on case 10 should score 0 on case 10 and the run should continue.
- **Write a report a human will read**, in markdown, into the pull request.
- **Save the report as JSON.** That is your baseline. A **regression** is a case that passed in the baseline and fails now, or an overall drop bigger than the noise you tolerate.
- **Two kinds of set.** The **golden set** is curated, versioned and run on every change; you look at its failures constantly. The **held-out set** is one nobody optimises against. The gap between them is the tell for contamination, which the boss will demonstrate.
- **Version the cases.** When a golden case changes, the baseline changes; record which version scored what.

### Goodhart's law

> *When a measure becomes a target, it ceases to be a good measure.*

Nothing about the metric changes when people start optimising it. What changes is the *population of systems* you are sampling from: it now includes every system that scores well *without doing the thing*. Concrete examples, all of which the Gorgon fields as champions:

| the metric | the gaming strategy | the second number that catches it |
|---|---|---|
| accuracy on 95/5 data | predict the majority class | balanced accuracy, per-class recall |
| pairwise win rate under an LLM judge | pad every answer with filler | length-controlled judging, filler ratio |
| score on the golden set | memorise the golden set | score on a held-out set; the gap |
| perplexity on a benchmark | train on the benchmark | contamination checks, fresh test sets |
| mean user rating | ask only happy users | response rate, rating distribution |
| latency p50 | drop the hard requests | p99, error rate |

The defence is always the same shape: a *second* metric that the first one's gaming strategy does not move, read next to the first. Never one number.

### Calibration (the secret room)

Accuracy says how often a model is right. **Calibration** says whether its confidence means anything: among predictions made with 90% confidence, about 90% should be right. Most neural networks are over-confident. **Expected calibration error** bins predictions by confidence and weights each bin's gap by how full it is:

```
ECE = Σ_b (n_b / N) · | accuracy_b − mean_confidence_b |
```

**Temperature scaling** fixes over-confidence with one scalar: divide the logits by T before the softmax, choosing T to minimise NLL on held-out data. T > 1 softens, T < 1 sharpens, and no argmax changes, so accuracy is untouched.

---

## Rooms

Run `dungeon enter 11` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code, then run that room's trial.

### 11.1 The Scales — `rooms/room_1_the_scales.py`

A brass balance the height of a door. The steward weighs your metrics against counts done by hand: a three-class confusion matrix, binary precision/recall/F1 with every zero-division case, macro-F1, balanced accuracy, and two perplexities. Then she hands you a champion with 95% accuracy and asks whether you are impressed.

```
dungeon trial 11 room_1
```

### 11.2 The Bootstrap Oracle — `rooms/room_2_bootstrap_oracle.py`

An oracle with a bag of numbered stones, drawing with replacement. `bootstrap_ci` for any statistic, `paired_bootstrap` for comparing two systems on the same items, `is_significant`, `required_sample_size`. The trial checks that the interval contains the truth, widens when the sample shrinks, and notices a small gain only when you resample in pairs.

Then **the Oracle's Prophecy**: three arena results (n = 50 at 80% vs 84%; n = 2000 at 80% vs 84%; n = 200 at 60% vs 75%). Write "distinguishable" or "not distinguishable" for each *before* running anything, reasoning from the standard error.

```
dungeon trial 11 room_2
```

### 11.3 The Judge — `rooms/room_3_the_judge.py`

A raised bench and a judge who has never been wrong about a contest with an obvious winner. Build `judge_prompt` and `parse_judgement` (six decorated replies to survive), `llm_judge`, `pairwise_prompt`, and `pairwise_judge_debiased`, which runs both orders and says "tie" when the judge contradicts herself. Then `cohens_kappa`, `average_ranks`, `spearman` and `agreement_report` against a labelled set of thirty answers.

The floor's mock judges read a hidden `[quality=N]` tag in each answer instead of reading the text. The prompt contract in the module docstring is how they find the candidates; follow it exactly.

```
dungeon trial 11 room_3
```

### 11.4 Golden Trials — `rooms/room_4_golden_trials.py`

A vault of 24 sealed scrolls about the dungeon (loot arithmetic, floor facts, shape questions), tagged three ways. Build `EvalCase`, the scorers, `EvalSuite.run`, `EvalReport.to_markdown`, JSON `save_report`/`load_report`, and `compare_to_baseline`. The honest champion scores 21/24. Then yesterday's champion comes back unable to count gold, and your regression check must name the seven scrolls that broke.

```
dungeon trial 11 room_4
```

---

## Boss: The Goodhart Gorgon

```
                  ,~~~~~~~~~,
                 (  s  s  s  )         "Show me the number you are proud of.
                 ( s (o)(o) s )         I will show you how to hit it without
                 (  s ______ s )        getting one bit better."
                  `~~~~~~~~~~~'
                 ___/ | || | \___        Her stare turns any metric into a
                /   ##########   \       TARGET. Three champions have already
               /   ############   \      looked: one predicts the majority class,
              |   ##  ARENA  ##    |     one pads every answer with 200 words of
              |___################_|     "furthermore", one memorised the golden
                                         scrolls and nothing else.
```

**Weakness:** metrics that are hard to game, and someone who reads more than one number at a time.

- **Phase 1:** `robust_classification_report`: accuracy beside balanced accuracy, macro-F1 and per-class recall, with a verdict that says "accuracy is lying" when the gap exceeds 0.2.
- **Phase 2:** `filler_ratio`, `truncate_to_words`, `length_controlled_judge`: the padded answer beats the concise correct one under the naive length-biased judge, and position debiasing alone does not save you. Truncate both candidates to the shorter one's word count (or judge under a rubric that penalises filler), then verify with a swap.
- **Phase 3:** `split_golden_heldout` (stratified by tag) and `contamination_check`: the memoriser's golden score is 1.0, its held-out score is 0.0, and the gap is the tell. The honest champion's gap is noise.
- **Phase 4:** `judging_report`, `GorgonReport.gamed_metrics`, and **the Gorgon's Prophecy**: which single number does each champion make look good?

```
dungeon fight 11
```

## Secret room: The Calibration Mirror *(optional)*

Behind a loose stone in the arena wall. `softmax` with a temperature, `mean_nll`, `reliability_table`, `expected_calibration_error`, `temperature_scale`. The trial builds a champion that is calibrated by construction and one that is three times too sure of itself; the mirror must measure the gap and a grid search over T must close it, without changing a single prediction.

```
dungeon trial 11 --secret
```

## Loot

Clear the four rooms and defeat the Gorgon to unlock:

- **Eval Harness** — `loot/eval_harness.py`. A standalone, importable copy of `EvalCase` / `EvalSuite` / `EvalReport` / `compare_to_baseline`, plus `bootstrap_ci`, `paired_bootstrap` and a swap-debiased pairwise judge. numpy and the standard library only.
- **Evaluation Cheat Sheet** — `loot/evaluation_cheat_sheet.md`. Every metric with its formula and when it lies, the confidence-interval rule of thumb, a judge design checklist, eval hygiene, and a Goodhart failure catalogue.

## Stuck?

- `dungeon hint 11 room_2` reveals one hint at a time (three per room).
- The trial failure messages quote the numbers they saw and the formula they expected. Read them before reading the code.
- `solutions/` exists. Use it the way you would use a walkthrough: after an honest attempt, and to compare, not to copy.

When `dungeon map` shows the Proving Grounds cleared, take the stairs. You now know whether the things you build work, which puts you ahead of most of the arena. Below, the machinery that runs the whole dungeon is humming, and it is slower than it needs to be.

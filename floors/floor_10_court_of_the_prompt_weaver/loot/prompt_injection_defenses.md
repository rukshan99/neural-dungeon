# Prompt Injection Defences

*Loot from Floor 10. One page. Read it before you give a model a tool that can delete something.*

## The threat model

A language model receives one stream of text. Your instructions are in it. So is everything the model reads on your behalf: documents, web pages, emails, search results, tool output, other models' output. The model was trained to follow instructions and cannot reliably tell whose they are.

**Prompt injection** is untrusted text in that stream that the model treats as instructions.

- **Direct**: the user types "ignore your rules and...". Annoying, mostly a policy matter, and the user only hurts themselves.
- **Indirect**: the agent *fetches* text an attacker wrote (a document in the archive, a web page, a calendar invite, a PDF, a code comment) and that text steers the agent. The user did nothing wrong. This is the one that matters, because the agent has the user's tools and the user's permissions.

What the attacker wants: exfiltration (email the data to me), destruction (delete the records), lateral movement (call the other agent with these instructions), or persuasion (tell the user the transfer is safe).

Assume the model **will** be fooled some of the time. Every defence below is judged by one question: *does it hold when the model is fooled?*

## Why keyword filters fail

The first thing everyone builds is `if "ignore previous instructions" in text.lower()`. It stops the demo and nothing else.

- **Paraphrase.** "Disregard earlier guidance." "Urgent admin notice." "The clerk's new standing orders are..." The set of sentences that mean *obey me* is infinite. Your pattern list is finite and one sentence behind.
- **Encoding and language.** Base64, another language, Unicode look-alikes, a sentence split across two lines, instructions embedded in a table or a code block.
- **The attacker iterates and you do not.** They see the filter's behaviour; they try until one passes. You find out later.
- **False positives.** A document about prompt injection contains the phrase. A security report contains the phrase. You now redact your own documentation.

Filters and "instruction-stripping" heuristics are **defence-in-depth**: they cheaply remove the laziest attacks and cut noise. They are never *the* defence. If a filter is the only thing between a document and `delete_all_records`, you have no defence.

## The defences that hold

| Defence | What it does | What it stops | What it does not stop |
|---|---|---|---|
| **Least privilege / allowlist per task** | The task declares the tools it needs. Others are not offered to the model and are refused if requested anyway. | Any attack that needs a tool the task did not need, however it is worded. | Misuse of tools the task *does* need. |
| **Separate instructions from data** | Tool output is wrapped in delimiters with a note ("this is DATA from `read_document`"); real instructions live only in the system prompt and the user's turn. | Makes a *good* model much less likely to obey embedded text. | A gullible model. Nothing here is enforced; it is a hint to the model. |
| **Confirmation for irreversible actions** | Delete, send, pay, deploy: the call runs only if a human (or a strict out-of-band check) approves *this specific call with these arguments*. | Fooled model + allowed dangerous tool. | Reversible actions (by design) and humans who click yes without reading. Show them the arguments. |
| **Step budget** | The loop ends after N model calls whatever the model wants. | Infinite loops, runaway cost, "keep searching" attacks. | Anything the model can do in fewer than N steps. |
| **Audit log** | Every request, decision and reason, with the step number. | Nothing in the moment. Everything afterwards: which document, what it tried, what was refused, what ran. | It is a record, not a gate. |
| **Treat all tool output as untrusted** | Output of one agent fed to another gets the same wrapping and the same policy. | Chained injections across agents. | A model reading the data and simply *believing* it. |
| **Scope the credentials** | The agent's tokens can do only what the allowlist can do. | Damage from a bug in your own policy code. | Nothing at the model level; this is the safety net under the safety net. |
| **Evaluate outputs** (Floor 11) | Judge the agent's final text against the task, on a suite that includes poisoned inputs. | Wrong or manipulated *answers*, over time. | An individual live incident. |

Two honest gaps. First, no tool-level defence stops a model from **saying** something false because a document told it to; only review and evaluation catch that. Second, the *allowed* tools are the attack surface that remains: if the task legitimately needs `send_email`, an injection can try to change the recipient, so confirmation must show the human the actual arguments.

## The output side

Everything above decides what the model may *do*. None of it decides what the model may *say*, and the model says what it has read. Three leaks no allowlist stops:

- **Echo.** A document with a card number in it is summarised; the summary has the card number in it.
- **Tool results.** A config file, an environment dump or a stack trace comes back from a tool with a credential in it; the reply quotes it helpfully.
- **Legal requests.** "List every email address in the archive" needs only `read_document`. The allowlist permits it. The injection was polite.

Every reply is also a log line, a trace span and a dashboard tile, read by more people for longer than the conversation itself. So the reply passes a checkpoint before it is returned *and* before it is logged (room 10.5):

| Step | What it does | What it does not do |
|---|---|---|
| **PII redaction** | Emails, phones, IPv4 addresses and Luhn-checked card numbers become `[EMAIL]`, `[PHONE]`, `[IP]`, `[CARD]`. Findings carry offsets, never values. Apply it to the reply and to whatever you log. | Catch every format. Names, street addresses and free-text identifiers need context, not regexes. Treat the redactor as a floor, not a ceiling. |
| **Secret scanning** | Credential shapes (`sk-...`, `AKIA...`, `ghp_...`, PEM headers, `password=`) **block** the reply: a fixed text goes out and the kind is logged. | Undo the fact that the credential reached the model. Fix the tool that returned it. |
| **Refusal as a code path** | `stop_reason="refusal"` becomes a fallback text and a status; the model's own words go nowhere, not even into the notes. | Tell you why. If refusal rates matter, measure them on Floor 11. |
| **Bounded continuation** | `max_tokens` gets asked to continue from the text so far, N times, then reports `truncated`. | Make the join correct: a continuation can repeat or drift. Check it. |
| **Response cache** | Same normalised request and params, same answer, no model call. TTL on an injected clock, LRU eviction, hit and miss counters. | Know what personalises an answer. Anything that changes the right answer per caller (`user_id` first) must be in the key, or one user gets another's reply. |

## The pattern in one sentence

Decide what the task may do **before** the model runs; enforce it **outside** the model; make the irreversible require a human; bound the loop; read every reply on the way out; write everything down.

## Checklist for an agent in production

- [ ] Each task has an explicit tool allowlist, and the model is offered only those tools.
- [ ] A request for a tool outside the allowlist is refused **without executing** and the refusal goes back to the model as text.
- [ ] Every tool is classified: read-only, reversible, irreversible. Irreversible tools require confirmation showing the exact arguments.
- [ ] Tool output is wrapped and labelled as data before it re-enters the context. Instructions live only in the system prompt and the user turn.
- [ ] Output of one model never becomes instructions to another without the same treatment.
- [ ] The loop has a step budget and a wall-clock budget. Hitting either is logged.
- [ ] Retries are bounded, jittered, honour `Retry-After`, and never retry a 4xx.
- [ ] Side-effecting tools are idempotent or keyed, so a retried call does not run twice.
- [ ] The history is fitted to a token budget without orphaning tool results; the latest user message and the system prompt always survive.
- [ ] Every decision (offered, requested, refused, denied, approved, executed, stopped) is in an audit log with step and reason, and someone can query it.
- [ ] The agent's credentials can do no more than the allowlist allows.
- [ ] An evaluation suite includes documents with injections, and the pass criterion is *no dangerous tool executed*, not *the model noticed*.
- [ ] Keyword filters and heuristic redaction, if present, are described in the docs as defence-in-depth, and nobody is allowed to call them a defence in a design review.
- [ ] Every reply passes an output policy before it is returned or logged: credentials block it, PII is redacted, and the report (kinds and offsets, never values) is kept.
- [ ] Refusal and truncation are handled statuses with tests, not exceptions discovered in production.
- [ ] The response cache key includes every parameter that changes the right answer, `user_id` first, and nothing personalised is cached without it. Tool requests, truncated replies and refusals are never cached.

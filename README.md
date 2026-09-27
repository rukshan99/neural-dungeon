# Neural Dungeon

```
 _   _                      _   ____
| \ | | ___ _   _ _ __ __ _| | |  _ \ _   _ _ __   __ _  ___  ___  _ __
|  \| |/ _ \ | | | '__/ _` | | | | | | | | | '_ \ / _` |/ _ \/ _ \| '_ \
| |\  |  __/ |_| | | | (_| | | | |_| | |_| | | | | (_| |  __/ (_) | | | |
|_| \_|\___|\__,_|_|  \__,_|_| |____/ \__,_|_| |_|\__, |\___|\___/|_| |_|
                                                   |___/
```

**A code-first dungeon crawl from AI engineering fundamentals to production.**

You are a software engineer. You can ship. You have used an LLM API and maybe trained a model by copying a notebook. What you do not have is the feeling that you *understand* the thing, from the gradient up to the KV cache. Neural Dungeon is fourteen floors of hands-on, test-driven exercises that build that understanding, wrapped in a game so you keep going.

- **Every lesson is code you write.** Rooms are Python files with `TODO`s. Trials are pytest suites that judge them. Nothing is cleared by reading.
- **Every floor is a topic.** Broadcasting. Gradient descent. Backprop from scratch. PyTorch. Tokenizers. Attention. A GPT you train yourself. LoRA, instruction tuning and DPO. Retrieval. Agents. Evals. Inference optimization. Monitoring what you deployed.
- **Every floor has a boss** whose weakness is the concept being tested. The Overfit Hydra grows a head for every memorized training example. The Oracle Who Peeks leaks the future through a bad causal mask.
- **Runs on a laptop CPU, offline, with no API keys.** A GPU is never required. Real-provider adapters exist for the LLM floors but every trial uses deterministic mocks.
- **Story is the wrapper, not the content.** The lore is there to make you smile; the engineering is precise. When the two conflict, the engineering wins.

## Quickstart

```bash
git clone https://github.com/rukshan99/neural-dungeon.git
cd neural-dungeon
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
dungeon doctor
dungeon enter 0
```

Python 3.11 or newer. Floors 0 to 3 need only numpy. From Floor 4 you need PyTorch; the CPU build is small and plenty:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(or `pip install -e ".[torch]"` for the default wheel on your platform).

## How to play

```
dungeon map                  the dungeon and your progress
dungeon enter 1              a floor's lore, rooms, boss, loot and status
dungeon trial 1              run every room trial on floor 1
dungeon trial 1 room_2       run one room's trial (also: dungeon trial 1 2)
dungeon fight 1              the boss fight
dungeon trial 1 --secret     the optional secret room
dungeon hint 1 room_2        reveal the next hint (three per room, one at a time)
dungeon loot                 cheat sheets and tools you have unlocked
dungeon status               your rank and numbers
dungeon reset                start over (asks first)
```

The loop is always the same:

1. `dungeon enter N` and read the floor's `README.md`. It teaches the concepts first, then describes each room as a quest.
2. Open `floors/<floor>/rooms/<room>.py`. Replace every `raise NotImplementedError` with code.
3. `dungeon trial N <room>`. Read the verdict. Failure messages are written to tell you *what* is wrong.
4. When every room is cleared, `dungeon fight N`. Defeating the boss clears the floor and unlocks its loot.
5. Take the stairs.

`dungeon trial` is plain pytest underneath; `pytest floors/floor_01_caverns_of_descent` works too, and so do `-k`, `-x`, `--pdb` and friends. Progress is stored in `.dungeon/progress.json`, which is git-ignored: it is your save file. A room counts as cleared only when its *whole* trial file passes in one run.

Every room has a reference implementation in `solutions/`. It is there so the repository can test itself and so you can compare after an honest attempt. Locks on loot are on the honour system.

## The map

| Floor | Name | You will learn | Boss | Needs |
|---|---|---|---|---|
| 0 | **The Threshold** | arrays, shapes, broadcasting rules, vectorization, einsum | The Broadcasting Basilisk | numpy |
| 1 | **The Caverns of Descent** | loss functions, gradients, gradient checking, gradient descent, momentum, Adam, schedules | The Learning-Rate Lich | numpy |
| 2 | **The Chain of Whispers** | the chain rule, a scalar and tensor autograd engine from scratch, vanishing gradients | The Vanishing Wraith | numpy |
| 3 | **The Forge of Layers** | MLPs, initialization, mini-batch training loops, validation, regularization | The Overfit Hydra | numpy |
| 4 | **The Torchlit Passage** | PyTorch tensors, autograd, `nn.Module`, `DataLoader`, debugging cursed training loops, device-agnostic code, mixed precision and loss scaling, determinism | The Reproducibility Revenant | torch |
| 5 | **The Scriptorium of Tokens** | character and byte-pair tokenizers, Unicode, embeddings, positional encodings, token budgets | The Babel Golem | torch |
| 6 | **The Hall of a Thousand Heads** | scaled dot-product attention, causal and padding masks, multi-head attention, leakage detection, rotary embeddings, grouped-query attention | The Oracle Who Peeks | torch |
| 7 | **The Tower of the Transformer** | LayerNorm and RMSNorm, transformer blocks, a GPT you train on the dungeon's own chronicles, sampling strategies, data-parallel training with all-reduce and DDP | The Stuttering Sovereign | torch |
| 8 | **The Vault of Frozen Weights** | checkpoints, freezing, LoRA from scratch, fine-tuning and catastrophic forgetting, instruction tuning with loss masking, direct preference optimisation | The Catastrophic Forgetter | torch |
| 9 | **The Library of Echoes** | embeddings, cosine similarity, vector indexes, chunking, RAG, grounding and citations | The Hallucinating Librarian | numpy |
| 10 | **The Court of the Prompt Weaver** | structured outputs, tool-calling loops, retries and backoff, context budgets, prompt injection defenses, output guardrails (PII, secrets, refusals) and response caching | The Injected Imp | none |
| 11 | **The Proving Grounds** | classification metrics, perplexity, bootstrap confidence intervals, LLM-as-judge, eval harnesses, regression | The Goodhart Gorgon | numpy |
| 12 | **The Engine Room** | KV caching, int8 quantization, batching, latency percentiles, a streaming inference server, speculative decoding | The Latency Leviathan | torch |
| 13 | **The Watchtower** | tracing, drift detection (PSI, KS, JS), canary releases and ramps, alerting with hysteresis, error budgets, the label-delay and peeking problems | The Silent Drift | numpy |

Fourteen floors, 66 rooms, 14 bosses, 14 secret rooms, 1,474 trials. Each floor takes an evening or two. Floors are independent enough to enter out of order if you already know a topic, but the first time through, go down in sequence: later floors reuse habits (and one shared tiny GPT) from earlier ones. When the map shows every floor cleared, read [EPILOGUE.md](EPILOGUE.md).

## What a floor looks like

```
floors/floor_01_caverns_of_descent/
├── README.md          the lesson: concepts first, then each room as a quest
├── floor.toml         name, lore, rooms, boss, secret room, loot
├── rooms/             YOUR files. Stubs with docstrings and TODOs.
├── trials/            pytest suites, one per room. Witty names, useful failure messages.
├── hints/             three progressive hints per room, revealed by `dungeon hint`
├── solutions/         reference implementations (spoilers)
└── loot/              cheat sheets and reusable snippets unlocked on clearing the floor
```

Rooms come in several formats so it never gets samey: implement-from-docstring, **prediction games** (write down the shape or the loss curve *before* running), **debug the cursed code** (find the seven planted bugs), **speedruns** (beat the loop by 20x), **build-then-break** (write the defense, then write the attack that gets past your own defense).

## Principles

- **Tests are the teacher.** A trial that fails should tell you what concept you are missing, not just that an assertion is false.
- **Offline and reproducible.** Synthetic datasets, a bundled corpus, seeded randomness, mock LLMs with deterministic behaviour. Nothing downloads, nothing costs money, nothing flakes.
- **No framework magic before you have built it yourself.** You write backprop before you call `.backward()`. You write attention before you call `nn.MultiheadAttention`. Then you use the library, gladly.
- **Production is part of the curriculum.** Determinism, evaluation, retries, injection defenses, KV caches and latency percentiles are not appendices. They are floors.
- **Inclusive humour.** The jokes are about shapes and gradients, never about people, and never require you to already be inside a particular culture or company.

## Scope

The path is deliberately the modern, LLM-centred one: from tensors to a transformer you train yourself, then to the systems built around such models (fine-tuning, retrieval, agents, evaluation, inference, operations). Inside that path it aims to be complete, including the production topics courses usually skip: mixed precision, data parallelism, instruction tuning and preference optimisation, injection and output guardrails, evals, KV caches, quantisation, monitoring and canary releases.

Outside that path, and not covered (yet): convolutional networks and computer vision, recurrent networks, reinforcement learning beyond preference optimisation, diffusion models, classical machine learning (trees, SVMs, clustering beyond k-means), and multi-node training at real scale. Each would make a fine floor; [CONTRIBUTING.md](CONTRIBUTING.md) is the blueprint.

## FAQ

**Do I need a GPU?** No. Every trial is sized for a laptop CPU. If you have one, PyTorch will use it where it helps, and nothing changes.

**Do I need an API key?** No. The LLM floors (9 to 11 and 13) use deterministic mock models so trials are fast and free. Optional adapters for real providers are included for your own experiments, never for trials.

**I am stuck.** `dungeon hint N room` gives one hint at a time. Failure messages are deliberate. The floor README covers every concept a room needs. After that, `solutions/`.

**Can I skip floors?** Yes. Nothing is locked. But the bosses assume you have the habits of the floors above them.

**How is progress tracked?** A pytest plugin records each trial file's result in `.dungeon/progress.json`. Delete the file or run `dungeon reset` to start again.

**Can I contribute a floor?** Yes, please. See [CONTRIBUTING.md](CONTRIBUTING.md) for the floor-authoring guide and the checks every floor must pass.

## For maintainers

```bash
DUNGEON_SOLUTIONS=1 pytest floors        # every trial must pass against the reference solutions
python scripts/check_stubs_fail.py       # every trial must fail against the untouched stubs
python scripts/lint_floors.py            # every manifest, room, trial, hint and loot path must agree
```

## License

MIT. See [LICENSE](LICENSE).

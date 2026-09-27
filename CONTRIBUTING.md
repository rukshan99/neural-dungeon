# Contributing to Neural Dungeon

Thank you for wanting to dig. This guide covers fixing rooms, tuning trials and, the big one, authoring a floor. It is also the exact specification the maintainers hold themselves to, so read it even if you only want to fix a typo in a boss's taunt.

## Ground rules

1. **The engineering is never wrong for the sake of a joke.** Lore wraps the content; it does not bend it. If a metaphor would mislead, cut the metaphor.
2. **Offline, CPU, deterministic.** No downloads, no API keys, no GPU requirement, no flaky timing. Seed every random generator. Timing assertions compare two implementations on the same machine, never against an absolute number.
3. **Tests are the teacher.** Every failure message says what concept is missing or which shape is wrong, in words a learner can act on.
4. **Inclusive humour.** Jokes are about tensors, gradients and the poor decisions of fictional adventurers. Never about people, groups, or references that need a specific culture or workplace to land.
5. **Small trials.** A trial file should finish in under ~20 seconds on a laptop CPU. Mark anything slower with `@pytest.mark.slow` and keep it out of the required path.

## Setup for development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Two checks must pass before anything merges:

```bash
DUNGEON_SOLUTIONS=1 pytest floors        # every trial passes against solutions/
python scripts/check_stubs_fail.py       # every trial fails against untouched rooms/
```

`ruff check .` keeps the style consistent.

## Anatomy of a floor

```
floors/floor_NN_snake_case_name/
├── __init__.py                 (empty; makes the floor importable)
├── floor.toml                  manifest: name, lore, rooms, boss, secret, loot
├── README.md                   the lesson + the quests
├── rooms/
│   ├── __init__.py
│   ├── room_1_<slug>.py        learner stubs
│   ├── room_2_<slug>.py
│   ├── boss_<slug>.py
│   └── secret_<slug>.py        optional hard challenge
├── solutions/
│   ├── __init__.py
│   └── <same file names as rooms/>
├── trials/
│   ├── __init__.py             (REQUIRED: floors share test file names)
│   ├── test_room_1.py
│   ├── test_room_2.py
│   ├── test_boss.py
│   └── test_secret.py
├── hints/
│   ├── room_1.md               3 hints, separated by a line containing only ---
│   ├── boss.md
│   └── secret.md
└── loot/
    └── <cheat_sheet>.md, <snippet>.py ...
```

Every `__init__.py` matters: trials import rooms as `floors.floor_NN_x.rooms.room_1_y`, and pytest needs the package structure to tell `floor_00/trials/test_room_1.py` from `floor_01/trials/test_room_1.py`.

Floors that need PyTorch declare `requires = ["torch"]` in `floor.toml` and start each trial file with:

```python
torch = pytest.importorskip("torch", reason="This floor needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
```

### floor.toml

```toml
id = "floor_01"                 # floor_NN, two digits
number = 1
slug = "caverns_of_descent"
name = "The Caverns of Descent"
tagline = "One italic sentence of atmosphere."
topics = ["loss functions", "gradients", "gradient descent"]
requires = []                   # or ["torch"]

map = '''
ASCII map of the floor's rooms. Keep it under 80 columns.
'''

[lines]                         # flavour the plugin prints; all optional
room_cleared = "..."
room_failed = "..."
room_cursed = "..."             # shown when a mode="cursed" room still has curses
boss_intro = "..."
boss_defeated = "..."
floor_cleared = "..."

[[rooms]]
id = "room_1"                   # room_N for regular rooms; "boss"; "secret"
name = "The Altar of Loss"
kind = "room"                   # room | boss | secret
file = "rooms/room_1_altar_of_loss.py"
trial = "trials/test_room_1.py"
blurb = "One or two sentences shown by `dungeon enter`."
# mode = "cursed"               # optional: a debug-the-cursed-code room that ships
                                # complete-but-buggy code; the summary then reads
                                # "curses lifted" instead of "failed"

[[rooms]]
id = "boss"
name = "The Learning-Rate Lich"
kind = "boss"
file = "rooms/boss_learning_rate_lich.py"
trial = "trials/test_boss.py"
blurb = "..."

[[rooms]]
id = "secret"
name = "..."
kind = "secret"
file = "rooms/secret_....py"
trial = "trials/test_secret.py"
blurb = "..."

[[loot]]
name = "Optimizer Grimoire"
file = "loot/optimizers.md"
blurb = "What it is and why the learner will want it."
```

A floor has 3 to 5 regular rooms, exactly one boss, and at most one secret room. Regular rooms plus the boss are required to clear the floor; the secret is a bonus.

## Writing a room stub (`rooms/`)

The stub is what the learner reads most, so it carries the lesson at the point of use.

```python
"""ROOM 1.1 - THE ALTAR OF LOSS

    Two or three lines of lore that set the scene.

Then the technical framing: what this room teaches, the key idea in plain
words, the pitfalls the trial will check. Link the concept to where it
reappears later in the dungeon when that is true.
"""

from __future__ import annotations

import numpy as np


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error over all elements. Return a Python float.

    Precise contract: shapes, dtypes, edge cases, what to return. Hints
    belong here only when they are part of the concept, not the answer.
    """
    raise NotImplementedError("mse() is unwritten")
```

Rules:

- Every stub body is exactly `raise NotImplementedError("<name>() is unwritten")` (the plugin shows these as *unwritten*, not *failed*, and `scrutiny.is_stub` detects them). Prediction-game dictionaries use `None` for unfilled answers.
- Type hints and docstrings state the *contract*: shapes, dtypes, return type, edge cases.
- Never import from `solutions/`. Rooms may import earlier rooms of the same floor with a relative import (`from .room_1_altar_of_loss import mse`); the same line works inside `solutions/`.
- Keep the file self-contained otherwise. Cross-floor code belongs in `dungeon/artifacts/`.

## Writing the solution (`solutions/`)

Same file name, same public names, fully working, clean, commented where the idea is subtle. Header: `(reference solution)` plus a spoiler warning. Solutions are also documentation; write them the way you would want to read them after struggling.

## Writing the trial (`trials/`)

```python
"""TRIAL 1.1 - THE ALTAR OF LOSS

One or two lines that say what is being judged.
"""

import numpy as np
import pytest

from dungeon.trials import load_room

room = load_room(__file__, "room_1_altar_of_loss")   # solutions/ when DUNGEON_SOLUTIONS=1


def test_mse_of_a_perfect_prediction_is_zero():
    assert room.mse(np.ones(4), np.ones(4)) == 0.0, "A perfect prediction has zero loss. Yours has some."
```

Rules:

- `load_room(__file__, "<module>")` always. It is the switch that lets CI run solutions.
- Test names read as narrative: `test_the_hydra_grows_a_head_for_every_memorized_example`. Failure messages explain the *concept* or *shape* that is wrong, with the observed value.
- Boss trials set `pytestmark = pytest.mark.boss`; secret trials set `pytest.mark.secret`.
- If a test would fail only because a prediction is still `None` or a function is still a stub, `raise NotImplementedError(...)` instead so the summary says *unwritten*.
- Seed everything (`np.random.default_rng(seed)`, `torch.manual_seed`). Tolerances generous enough for float32 on any CPU.
- Timing: measure a slow reference and the learner's version on the same input, assert a ratio. Never assert absolute seconds.
- Use `dungeon.scrutiny` (`python_loops_in`, `names_called_in`, `operators_used_in`, `is_stub`) when a room forbids or requires a construct.
- Trials may monkeypatch library functions to enforce a rule (the Basilisk petrifies `np.broadcast_shapes`). Do it through the `monkeypatch` fixture so it is undone.
- One trial file per room; the file must not pass against the stub (the check script enforces this).
- A floor may add `trials/conftest.py` for floor-wide fixtures (Floor 4 pins torch to one thread because its models are tiny). The root `conftest.py` already caps BLAS/OpenMP pools at 4 threads before numpy or torch load; measure before changing that for your floor, since the 800K-parameter Chronicler is faster with several threads.
- Shared non-stub fixtures (a corpus, a mock model, tool implementations) live in `<floor>/assets/` with an `__init__.py`. Assets never import from `rooms/` or `solutions/`; trials build learner objects from asset data instead, so a broken room cannot break the fixtures.
- If a stub imports a name only so its docstring can refer to it, mark the import `# noqa: F401`; `ruff --fix` would otherwise strip it.

## Hints (`hints/<room_id>.md`)

Three hints, separated by a line containing only `---`. Hint 1 points at the idea. Hint 2 points at the tool or the shape. Hint 3 is nearly the code. The CLI reveals one per invocation and records how many were used.

## Loot (`loot/`)

Something the learner will *reuse*: a one-page cheat sheet, a clean reference implementation, a checklist, a template. Loot is the reward for the boss, so it should feel like one.

## The floor README

The README is the lesson. Structure:

1. Title, italic tagline, the ASCII map (same as `floor.toml`).
2. Two or three paragraphs of lore that also state, honestly, why this topic matters to an engineer.
3. **You will learn** and **You need**.
4. **The lore of X**: the concepts, taught properly. Equations where they help, code where it is clearer, both when needed. This section should let a strong engineer with no ML background do every room.
5. **Rooms**: one subsection per room. Story hook, exact task, the command to run.
6. **Boss**: ASCII art, a taunt, the **Weakness** line (the concept), the phases, the command.
7. **Secret room** (optional) and **Loot**.
8. **Stuck?** pointing at hints, failure messages and solutions.

Accuracy beats atmosphere. If you are not sure a claim is correct, check it or cut it.

## Voice

Playful, dry, welcoming. Second person. Short sentences. The dungeon has a sense of humour about how hard this is and no contempt for anyone finding it hard. Bosses are allowed to be smug; the narrator is not.

## Submitting

- One floor or one fix per pull request.
- Run both checks and `ruff check .`.
- Describe what a learner will experience, not just what changed.

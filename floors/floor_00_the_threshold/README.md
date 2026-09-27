# Floor 0 — The Threshold

> *Every spell cast below this point is spoken in shapes. Learn the tongue before you descend.*

```
            ☀ the surface
            │
   ┌────────┴────────┐     ┌──────────────────┐     ┌──────────────────────┐
   │  0.1 THE GATE-  │─────│  0.2 THE HALL    │─────│  0.3 THE BROADCASTING│
   │     KEEPER      │     │     OF SHAPES    │     │        BRIDGE        │
   └─────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                               │
   ┌─────────────────┐     ┌──────────────────┐                │
   │ ☠ THE BASILISK'S│─────│  0.4 THE SPEEDRUN│────────────────┘
   │      LAIR       │     │     GALLERY      │
   └────────┬────────┘     └──────────────────┘
            ┆  ◇ a draught from below: the Einsum Oubliette
            ▼  stairs down to Floor 1
```

You stand at the mouth of the Neural Dungeon. Behind you: the surface, where numbers are scalars and loops are fine. Ahead: an archway guarded by a rusted automaton that speaks only in shapes.

Everything below this floor is built from arrays. Neural networks are matrix multiplications with a nonlinearity between them. Attention is a batched matrix product followed by a softmax. Every bug you will ever chase in a training loop is, about half the time, a shape bug. The Threshold exists so that you never again have to think about the *plumbing* while you are trying to think about the *ideas*.

**You will learn:** arrays and dtypes · shapes, axes and reshaping · the broadcasting rules · vectorization · `einsum`.

**You need:** Python 3.11+ and numpy. No PyTorch until Floor 4.

---

## The lore of arrays (read this before the rooms)

### Arrays have a shape, a dtype, and a memory layout

A numpy array is a flat buffer of memory plus metadata that says how to interpret it: `shape` (how many elements along each axis), `dtype` (what each element is), and `strides` (how many bytes to step to move along each axis). Almost every operation in this floor changes the metadata without touching the buffer.

```python
x = np.zeros((2, 3, 4), dtype=np.float32)
x.shape   # (2, 3, 4)   three axes: 0, 1, 2
x.ndim    # 3
x.size    # 24          total elements
x.dtype   # float32
```

numpy defaults to `float64`. Deep learning defaults to `float32` and smaller. Being explicit about `dtype=` is the first habit the dungeon asks of you.

### Axes are the language of reductions

"Sum over axis 1" means: axis 1 disappears, everything else stays.

```python
x = np.zeros((2, 3, 4))
x.sum(axis=1).shape                 # (2, 4)
x.sum(axis=1, keepdims=True).shape  # (2, 1, 4)   the axis stays as a 1
x.mean(axis=(0, 2)).shape           # (3,)
x.max(axis=-1).shape                # (2, 3)      -1 is always "the last axis"
```

`keepdims=True` is not cosmetic. It is the difference between a result that broadcasts back against the input and one that does not.

### Reshape reinterprets; transpose reorders

- `reshape` keeps the elements in the same order and regroups them. It never reorders anything, and on a contiguous array it never copies. `x.reshape(B, -1)` lets numpy compute one dimension.
- `transpose` changes which axis comes first. The elements you see move.

These are different operations that both change `shape`. Using `reshape` where you needed `transpose` produces no error and scrambled data. Channels-last `(B, H, W, C)` to channels-first `(B, C, H, W)` is `x.transpose(0, 3, 1, 2)`. Always.

### Indexing can add or drop axes

```python
x = np.zeros((5, 4))
x[:, -1].shape    # (5,)   the last column, one dim dropped
x[:, -1:].shape   # (5, 1) the last column, dim kept
x[None].shape     # (1, 5, 4) a new leading axis
x[:, None].shape  # (5, 1, 4) a new axis at position 1
```

`None` (alias `np.newaxis`) inserts a size-1 axis. You will use this constantly to set up broadcasting.

### Broadcasting: the four rules

Broadcasting is how numpy combines arrays of different shapes *without copying*. Memorize these; the boss checks.

1. **Align the shapes at the right edge.** Pad the shorter with 1s on the left.
2. **Walk the dims pairwise.** Each pair must be equal, or one must be 1.
3. **A 1 is stretched** to match the other dim. The result takes the larger.
4. **Anything else is an error.**

```
(8, 1, 6, 1)
(   7, 1, 5)    aligned at the right, padded with a 1
------------
(8, 7, 6, 5)    every column: equal or one is 1
```

Failure modes you will meet: `(N,) - (N, 1)` broadcasts to `(N, N)` silently, and dividing `(N, D)` by a `(N,)` vector *fails* unless `N == D`, in which case it silently divides the wrong axis. Both are cured by inserting axes deliberately.

### Vectorization: why loops are slow

Every iteration of a Python loop pays for bytecode dispatch, dynamic type checks and boxing floats into objects. numpy runs the same loop in compiled C over a contiguous buffer, usually with SIMD instructions, and usually 50–500x faster. The skill is recognizing the array operation hiding inside a loop: fancy indexing, `cumsum`, `keepdims`, a matrix product.

### Einsum: name the axes, say what to sum

`np.einsum("bqd,bkd->bqk", q, k)` reads: input 1 has axes `b, q, d`; input 2 has `b, k, d`; multiply where letters match, and sum over any letter that does not appear in the output (`d`). It is the clearest way to write attention scores, batched matmuls, traces and bilinear forms, and it appears again on Floor 6.

---

## Rooms

Run `dungeon enter 0` to see your progress. Each room is a file in `rooms/`. Replace every `raise NotImplementedError` with code, then run that room's trial.

### 0.1 The Gatekeeper — `rooms/room_1_the_gatekeeper.py`

A rusted automaton blocks the archway. It does not want a password; it wants to know your tools work and that you understand the loop: read the docstring, write the code, run the trial, read the verdict.

Three tiny functions. Notice the dtype requirement. It is the first of many.

```
dungeon trial 0 room_1
```

### 0.2 The Hall of Shapes — `rooms/room_2_hall_of_shapes.py`

Every statue has been moved to the wrong plinth overnight. Six functions to put them back: flatten a batch, move channels first, stack pairs, slice a column, take every other row, add a batch axis.

Then **the Prophecy**: ten expressions and their output shapes, which you must write *before* running anything. Committing to a prediction and being wrong teaches more than running it and nodding.

```
dungeon trial 0 room_2
```

### 0.3 The Broadcasting Bridge — `rooms/room_3_broadcasting_bridge.py`

A rope bridge over a chasm, held up by broadcasting alone. Seven functions: center rows, normalize rows, standardize columns, an outer sum, pairwise distances, per-channel scaling, per-column clamping.

**No loops.** No `for`, no `while`, no comprehensions. The trial reads your source and refuses them. Every function is one or two lines once you see the axis to insert.

```
dungeon trial 0 room_3
```

### 0.4 The Speedrun Gallery — `rooms/room_4_speedrun_gallery.py`

Four paintings of a tiny figure running the same corridor forever. The corridor is a Python loop. The trial file contains the slow loops; you write the fast versions. Correctness first, then a race: yours must be at least 10x faster than the loop *on your machine*.

One of the four (`softmax_rows`) is also your first meeting with numerical stability: `exp(1000.0)` is `inf`, and the fix (subtract the row max) will follow you down every floor with a softmax in it.

```
dungeon trial 0 room_4
```

---

## Boss: The Broadcasting Basilisk

```
                 .-~~~-.
               .'  o o  `.        "You lean on the machine to tell you
              /  .-'''-.  \        which shapes fit. Lean, then, and fall."
             |  /  ___  \  |
             | |  (o o)  | |         Its gaze PETRIFIES numpy's broadcasting
             |  \  `-'  /  |         helpers. During the fight, broadcast_shapes,
              \  `-...-'  /          broadcast_to, broadcast_arrays and the array
               `.  ~~~  .'           constructors all turn to stone. Some of its
              ,-'`-...-'`-.          eyes have 10**15 elements: you cannot build
             /   (ssss)    \         them to look.
```

**Weakness:** someone who knows the four rules and can apply them to plain tuples.

- **Phase 1:** `broadcast_shape(*shapes)`: implement the rules yourself, for any number of shapes.
- **Phase 2:** `along_axis`, `scale_along`, `petrified`: insert size-1 axes exactly where you mean to.
- **Phase 3:** `BASILISK_RIDDLES`: ten broadcasts to predict, some of which are errors.

```
dungeon fight 0
```

## Secret room: The Einsum Oubliette *(optional)*

A trapdoor under the Basilisk's lair. Seven operations; one permitted incantation. The trial reads your source and allows only `einsum`. Cleared or not, it does not affect whether the floor counts as done, but the loot cheat sheet's einsum table will make a lot more sense afterwards.

```
dungeon trial 0 --secret
```

## Loot

Clear the four rooms and defeat the Basilisk to unlock:

- **Grimoire of Shapes and Broadcasting** — `loot/shapes_and_broadcasting.md`. The rules, the axis-insertion table, reshape-vs-transpose, einsum idioms and the five shape bugs everyone hits.

## Stuck?

- `dungeon hint 0 room_3` reveals one hint at a time (three per room).
- The trial failure messages are written to tell you *what* went wrong, not just *that* it did. Read them.
- `solutions/` exists. Use it the way you would use a walkthrough: after an honest attempt, and to compare, not to copy.

When `dungeon map` shows the Threshold cleared, take the stairs. Below, someone is measuring loss.

Phase 1 is pure Python on tuples. Find the longest shape, left-pad every shape with 1s to that length: `(1,) * (ndim - len(s)) + tuple(s)`. Then `zip(*padded)` walks the dimensions position by position.
---
At each position, collect the dims that are not 1 into a set. If that set has more than one element, the shapes conflict: return None. Otherwise the result dim is the single non-1 value, or 1 if the set is empty.
---
Phase 2: `along_axis` builds `shape = [1] * ndim; shape[axis % ndim] = len(v)` and reshapes. `scale_along` is `x * along_axis(scale, x.ndim, axis)`. `petrified` is `np.where(mask[..., None], 0.0, x)`: the `None` adds the missing last axis so the mask broadcasts over D. Phase 3: apply your own phase-1 function in your head. (3,) vs (4,) is an error; (2,1) vs (8,4,3) fails at the 2 vs 4.

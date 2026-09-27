Every operator does the same four things: wrap a plain number (`other = other if isinstance(other, Value) else Value(other)`), build `out = Value(result, (self, other), "op")`, define a closure `_backward` that does `self.grad += <local derivative> * out.grad` for each child, and assign `out._backward = _backward`. The closure reads `out.grad` *later*, at backward time, when the parent has already filled it in. That is why `+=` and the order both matter.
---
Local derivatives: `+` gives 1 to both children; `*` gives the *other* operand's data; `x ** n` gives `n * x ** (n - 1)`; `exp` gives `out.data` (the output you already computed); `log` gives `1 / x`; `tanh` gives `1 - t * t`; `relu` gives `1.0 if x > 0 else 0.0`; `sigmoid` gives `s * (1 - s)`. The reflected operators are one-liners: `__radd__` returns `self + other`, `__rsub__` returns `(-self) + other`, `__rtruediv__` returns `self ** -1 * other`. `__neg__` is `self * -1`, `__sub__` is `self + (-other)`, `__truediv__` is `self * other ** -1`.
---
`backward()` is a depth-first post-order walk followed by a reversed sweep:

```python
topo, visited = [], set()
def build(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build(child)
        topo.append(v)          # only after all children
build(self)
self.grad = 1.0
for v in reversed(topo):
    v._backward()
```

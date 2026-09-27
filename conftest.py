"""Root pytest configuration for Neural Dungeon.

Registers the dungeon plugin, which records which trials you have cleared in
``.dungeon/progress.json`` and narrates the outcome after each run.

It also caps BLAS/OpenMP thread pools *before* numpy or torch are imported. The
dungeon's models are tiny; on a many-core machine a 22-thread matmul over a
64x64 matrix spends longer synchronising than multiplying, and trials that take
a second single-threaded were measured taking five. Four threads is a good
middle ground for every floor. Set the variables yourself to override.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

pytest_plugins = ["dungeon.trials"]

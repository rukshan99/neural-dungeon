"""Trial-wide settings for the Torchlit Passage.

The models on this floor are tiny (64 inputs, a few thousand weights). With many
intra-op threads torch spends longer synchronising than multiplying, and on a
busy machine that turns a 50 ms epoch into several seconds. Every trial on this
floor therefore runs torch single-threaded and restores the previous setting
afterwards.

Bitwise reproducibility (the boss) also assumes a fixed thread count: the order
in which partial sums are combined depends on it, and so do the last bits.
"""

import pytest

try:
    import torch
except ImportError:  # the trial files skip themselves with the install hint
    torch = None


@pytest.fixture(autouse=True)
def _single_threaded_torch():
    if torch is None:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)

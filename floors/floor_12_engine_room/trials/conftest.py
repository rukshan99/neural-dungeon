"""Trial-wide settings for the Engine Room.

Every timing trial on this floor compares two ways of running the same tiny
model on the same machine, and a ratio is only fair when both sides use the
same number of threads. The floor therefore runs torch single-threaded during
each trial and restores the previous setting afterwards, so that pinning the
thread count here never leaks into the floors above when the whole dungeon is
run in one session (their Chronicler trainings are faster with several threads).
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

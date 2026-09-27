"""Neural Dungeon: the game engine.

This package is the *wrapper*, not the lesson. It knows how to:

- find the floors of the dungeon (``registry``),
- remember which trials you have cleared (``progress``),
- run trials and narrate the outcome (``trials``, a pytest plugin),
- draw maps and talk to you in colour (``ui``, ``cli``).

None of the machine learning lives here. That is all in ``floors/``.
"""

__version__ = "0.2.0"

"""Root pytest configuration for Neural Dungeon.

Registers the dungeon plugin, which records which trials you have cleared in
``.dungeon/progress.json`` and narrates the outcome after each run.
"""

pytest_plugins = ["dungeon.trials"]

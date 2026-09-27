"""Allow ``python -m dungeon`` as an alias for the ``dungeon`` command."""

from dungeon.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

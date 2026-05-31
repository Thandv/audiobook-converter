"""Allow `python -m audiobook` and serve as the PyInstaller entrypoint."""
from .cli import cli

if __name__ == "__main__":
    cli()

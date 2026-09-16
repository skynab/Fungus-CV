"""PyInstaller entry point for the standalone Fungus-CV app."""

import multiprocessing
import sys

from fungus_cv.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed on Windows for frozen apps
    sys.exit(main())

"""Standalone A0 entry point, leaving the original v1 launcher unchanged."""
import runpy

from run import load_package


if __name__ == "__main__":
    runpy.run_module(f"{load_package()}.a0_runner", run_name="__main__")

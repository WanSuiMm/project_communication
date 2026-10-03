"""Focused CPU qualification for the causal follow-up intervention helpers."""
import json

from causal import cpu_check


def run_checks():
    result = cpu_check()
    if result.get("status") != "PASS" or result.get("training") is not False:
        raise AssertionError("Causal CPU qualification did not pass")
    return result


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))

"""CPU-only sanity checks for saved-case report aggregation."""
from __future__ import annotations

from analyze import check_additive_fixture, check_baseline_suffix_fixture


def main() -> None:
    check_additive_fixture()
    check_baseline_suffix_fixture()
    print("saved-case analyzer fixtures: PASS")


if __name__ == "__main__":
    main()

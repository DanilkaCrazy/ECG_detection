"""DEPRECATED — Stage 3.6.4b external test (known-buggy protocol).

This module is kept only for reproducibility of the *old* numbers.  It must
NOT be used as evidence of external validity: it trained on all Chapman rows
(including the internal test) and picked the threshold from the external set's
own labels.

Use ``scripts/stage369_external_test.py`` (leak-free protocol) instead.
"""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    print("This protocol is deprecated and methodologically invalid.")
    print("Run: python scripts/stage369_external_test.py")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

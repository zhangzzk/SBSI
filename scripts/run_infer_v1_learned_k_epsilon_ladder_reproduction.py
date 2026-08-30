#!/usr/bin/env python
"""Launch the exact recovered K/epsilon ladder through its minimal legacy loader."""

from __future__ import annotations

import sys
from types import ModuleType

from infer_v1_legacy_pipeline import _load_pipeline, _sync_cuda


def main() -> None:
    compatibility = ModuleType("test_infer_v1_proposal_diversification")
    compatibility._load_pipeline = _load_pipeline
    compatibility._sync_cuda = _sync_cuda
    sys.modules[compatibility.__name__] = compatibility

    from test_infer_v1_learned_k_epsilon_ladder import main as ladder_main

    ladder_main()


if __name__ == "__main__":
    main()

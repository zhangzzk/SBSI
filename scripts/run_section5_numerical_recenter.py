#!/usr/bin/env python
"""Temporary compatibility entry point for active 2026-08-29 Slurm jobs."""

from pathlib import Path
import runpy


runpy.run_path(Path(__file__).with_name("run_inference.py"), run_name="__main__")

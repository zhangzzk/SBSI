"""Runnable SBSI pipeline entrypoints.

These modules are scripts first (each has a ``main()`` behind an ``if __name__ ==
"__main__"`` guard), but several of them are also imported by their siblings to reuse a
data pipeline or model builder without reimplementing it -- e.g.
``from scripts.train_joint_forward import build_model``.  This file makes ``scripts`` an
explicit package so those imports resolve from the repository root rather than depending
on ``scripts/`` itself being on ``sys.path``.
"""

"""FIXTURE, not production code.  The same bug, at MODULE scope instead of in a function.

The sweep read function bodies only until 2026-08-19, which made module scope the one place
this defect could sit and never be reported.  A checker is silent on an unscanned region in
exactly the way it is silent on clean code, so nothing distinguished "no hits here" from
"never looked here".  This file is what makes the difference visible: it must fire.

The nested function below must be attributed to `helper`, NOT to `<module>` -- with the
module in the scope list, a plain ast.walk would report every nested loop twice.
"""
PRE = "preprocessor"
OUT = []
for m in range(3):
    ctx = PRE.upper() + str(m)
    OUT.append(ctx)
    PRE = OUT[:m + 1]                # <-- the bug, at module scope


def helper(seed, n):
    acc = seed
    got = []
    for i in range(n):
        got.append(acc * i)
        acc = got                    # <-- the same bug, one scope down
    return got

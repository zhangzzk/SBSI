"""How much blend response does a FAINT neighbour contribute? The emulator ignores neighbours
below its r<28 cut, but superposition says every neighbour still adds its R_blend term. If faint
neighbours contribute non-trivially, the emulator UNDER-counts R_blend -> positive m (matches gold
sign + the cg_ood result: +6.5%->+3.4% when OOD-neighbour galaxies are removed).

R_blend(probe @1.2") vs probe flux ratio (probe faint relative to a flux=1200 target).
"""
import sys
import numpy as np

sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI")
from scripts.toy_blend_decompose import resp  # noqa

TF, SKY, G, N = 1200.0, 10.0, 0.05, 300


def tgt():
    return dict(hlr=0.4, flux=TF, e1=0, e2=0, x=0, y=0, shear=False)


def probe(fr):
    return dict(hlr=0.4, flux=TF * fr, e1=0, e2=0, x=1.2, y=0.0, shear=True)


def main():
    print(f"[faint-nbr] target flux={TF}, probe @1.2\" on +x, shear ONLY probe")
    print(f"{'flux ratio':>11} {'mags fainter':>13} {'R_blend':>10} {'+/-':>7}")
    for fr in [1.0, 0.5, 0.25, 0.1, 0.04]:
        Rb, eb = resp([tgt(), probe(fr)], G, SKY, N)
        dmag = -2.5 * np.log10(fr)
        print(f"{fr:>11.2f} {dmag:>13.2f} {Rb:>10.4f} {eb:>7.4f}", flush=True)


if __name__ == "__main__":
    main()

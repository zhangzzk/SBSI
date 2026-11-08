"""Frozen helper functions from the existing GalSBI constant-shear evaluator.
No archived launcher or path injection; functions execute in the installed F8 runtime.
"""
import hashlib,json
from pathlib import Path
import numpy as np
REPO=Path('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe')
DATA=Path('/project/ls-gruen/users/zekang.zhang')
SIMS=DATA/'sbsi/galsbi_resim/constant_g002/sims'
PRESET=REPO/'configs/models_galsbi_dev.json'
OUT=DATA/'sbsi_caches/galsbi_dev_constant_20261103'
TILE='tile180.0_-0.5'
K,R_MAX=20,10.
CUTS=(('uncut', None, None), ('mag25.8_radius_gt0.60', ('>', 3.0), 25.8), ('mag25.8_radius_ge0.75', ('>=', 3.75), 25.8), ('mag25_radius_gt0.60', ('>', 3.0), 25.0), ('radius_gt0.60', ('>', 3.0), None), ('radius_ge0.75', ('>=', 3.75), None), ('mag25.8', None, 25.8), ('mag25', None, 25.0))
CUT_NAMES=tuple(c[0] for c in CUTS)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 23), b''):
            h.update(block)
    return h.hexdigest()

def output_cuts(values):
    """Keep mask (..., cut) for measured (numpy) or generated (torch) joint outputs: radius in pixels, flux."""
    radius, flux = values[..., 2], values[..., 3]
    rows = []
    for _, size, mag in CUTS:
        keep = radius == radius   # all true for finite rows; nonfinite measured rows are not usable anyway
        if size is not None:
            keep = keep & ((radius > size[1]) if size[0] == '>' else (radius >= size[1]))
        if mag is not None:
            keep = keep & (flux > 10**(-0.4*(mag-30)))
        rows.append(keep)
    return (np if isinstance(values, np.ndarray) else __import__('torch')).stack(rows, -1)

def draw_moments(keep, values, shift):
    """Selected mass, flow shape and disk shift over draws (as the frozen selected_draw_moments)."""
    keep = keep.to(values.dtype)
    return (keep.mean(1), (keep[..., None]*values[..., None, :2]).mean(1),
            (keep[..., None]*shift[:, :, None, :]).mean(1))

def preset():
    spec = json.loads(PRESET.read_text())
    paths = {'flow': DATA/'sbsi_caches'/spec['flow']['path'],
             'classifier': DATA/'sbsi_caches'/spec['classifier']['path'],
             'emulator': DATA/'blendemu_runs'/spec['emulator']['path']}
    for key, digest in (('flow', spec['flow']['sha256']), ('classifier', spec['classifier']['sha256']),
                        ('emulator', spec['emulator']['sha256'])):
        if sha256(paths[key]) != digest:
            raise ValueError(f'{key} artifact differs from the GalSBI-dev pin')
    conditions = {k: spec['observing_conditions'][k]
                  for k in ('pixel_size', 'zero_point', 'psf_fwhm', 'moffat_beta', 'pixel_rms')}
    return spec, paths, conditions

def full_context(context):
    """Pad to the 11 frozen FULL_FLOW_FEATURES columns; the unused FS2 nbr_flux_* columns are zero.

    Both frozen checkouts build sheared frames on these columns; F8 and C8 read their eight inputs by name."""
    full = np.zeros((len(context), 11))
    full[:, :5], full[:, 8:] = context[:, :5], context[:, 5:]
    return full

def truth(case, shear):
    import pandas as pd
    path = SIMS/f'case{case}_{shear:.2f}/real0/catalogues/input/gals_info_{TILE}.feather'
    frame = pd.read_feather(path).sort_values('index_input', kind='stable').reset_index(drop=True)
    return frame, path

def pair_table(scene, parent_rows, conditions):
    """Up to K nearest rendered neighbours within R_MAX of each parent row, scaled emulator features."""
    from sbsi.forward_catalogue import make_pair_catalogue, rescale_emulator_pairs
    from sbsi.output_conditioned_response import TRUTH_FEATURES
    pairs = make_pair_catalogue(scene.iloc[parent_rows], scene, r_max_arcsec=R_MAX, k=K)
    pairs = rescale_emulator_pairs(pairs, conditions)
    pairs = pairs[['index_input_p', 'index_input_s', *TRUTH_FEATURES]]
    pairs = pairs.sort_values(['index_input_p', 'index_input_s'], kind='stable').reset_index(drop=True)
    if pairs.duplicated(['index_input_p', 'index_input_s']).any() or not np.isfinite(
            pairs[list(TRUTH_FEATURES)].to_numpy()).all():
        raise ValueError('invalid emulator pairs')
    return pairs

def scene_frame(t):
    import pandas as pd
    return pd.DataFrame({name.replace('_input', ''): t[name] for name in t.columns})

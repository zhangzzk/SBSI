# Simulation-Based Shear Inference (SBSI)

![Constgold galaxy stamps and conditional measurement-flow contours](examples/measurement_flow_contours.png)

Flow likelihood for three example galaxies. Shape measurements are also contaminated by blending, which is in practice adjusted by a separate emulator.

---

SBSI is an integrated weak gravitational lensing (shear) calibration framework, accounting for detection, selection, and blending.

The [inference tutorial notebook](examples/sbsi_api_tutorial.ipynb) is the main prediction walkthrough; it uses the bundled V3 model.

## Installation

```bash
python -m pip install -e /path/to/SBSI
```

BlendEMU is only required for emulator prediction; the flow checkpoints need SBSI and
PyTorch alone. Clone [BlendEMU](https://github.com/zhangzzk/blendemu) and tell SBSI where it is:

```bash
git clone https://github.com/zhangzzk/blendemu.git
export BLENDEMU_ROOT=/path/to/blendemu
```

## Image simulation and measurement

SBSI training data are generated using BlendEMU. 
BlendEMU owns rendering, measurement, and simulation-catalogue construction.
SBSI calls its supported pipeline CLI. The
[example Slurm wrapper](examples/job_blendemu.sh) runs BlendEMU steps
1 through 4b from a user-owned YAML configuration. 

## Models

Model names are path presets (`sbsi.get_model(name)`); `python -m sbsi show-model <name>`
prints one and `python -m sbsi validate-model <name>` checks its files and hashes.

- **V3** — the tutorial model. Its 16 flow checkpoints and blending emulator are bundled in
  [`models/`](models/README.md), so it works from a fresh clone.
- **V3.9-like** — the current model: one measurement flow, one usable-detection
  classifier and a disk-shaped blending-response emulator. Its identity (paths and SHA-256
  hashes) is pinned in [`configs/models_v3_9_like.json`](configs/models_v3_9_like.json). The
  artifacts are not bundled; point `SBSI_CACHE_DIR` (flow, classifier) and
  `BLENDEMU_RUNS_DIR` (emulator) at the model store. On the development constant-shear
  simulations its multiplicative bias is m = +0.02% ± 0.27% (68%), after cuts of radius
  > 0.6 arcsec and magnitude < 25.8; it has not been independently validated.
- **V3.5-like**, **V3.6-like** — earlier presets kept so older results stay reproducible.

The disk emulator predicts a response per flow draw, so V3.9-like has its own short API
instead of the additive `load_emulator` path that V3 uses:

```python
inputs, pairs = sbsi.prepare_disk_inputs(catalogue, observing_conditions)
predictor = sbsi.DiskResponsePredictor(sbsi.get_model("V3.9-like"))
prediction = predictor.predict(inputs, pairs)   # +-0.02 shear legs, 64 paired draws
prediction.summary()                             # R_flow, R_blend, R_model with 1-sigma errors
```

The [V3.9 tutorial](examples/sbsi_api_tutorial_v39.ipynb) walks through it.

Flow, classifier and emulator training are not part of this release. Emulator training
lives in BlendEMU.

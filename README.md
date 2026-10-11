# Simulation-Based Shear Inference (SBSI)

![Constgold galaxy stamps and conditional measurement-flow contours](examples/measurement_flow_contours.png)

Flow likelihood for three example galaxies. Shape measurements are also contaminated by blending, which is in practice adjusted by a separate emulator.

---

SBSI is an integrated weak gravitational lensing (shear) calibration framework, accounting for detection, selection, and blending.

The [inference tutorial notebook](examples/sbsi_api_tutorial.ipynb) is the main prediction walkthrough; it uses the bundled V3.9-like model.

## Installation

```bash
python -m pip install -e /path/to/SBSI
```

Prediction with the bundled model needs SBSI alone. BlendEMU is needed only to simulate
new training data (and for the tutorial's optional import check). Clone [BlendEMU](https://github.com/zhangzzk/blendemu) and tell SBSI where it is:

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

- **V3.9-like** — one measurement flow, one classifier for "detected with a usable
  measurement", and a blending-response emulator applied to every flow draw. The files are
  bundled in [`models/`](models/README.md), so it works from a fresh clone; their SHA-256
  hashes are pinned in [`configs/models_v3_9_like.json`](configs/models_v3_9_like.json).

The tutorial's `ResponsePredictor` and `predict_blend_response` give the per-galaxy
response with no measurement cuts or detection weighting. The number validated on the
development constant-shear simulations, m = +0.02% ± 0.27% (68%), keeps galaxies with
measured radius above 0.6 arcsec and magnitude below 25.8, and weights each galaxy by its usable-detection
probability; the emulator returns it directly:

```python
emulator = sbsi.load_emulator(models, conditions=observing_conditions)
emulator.predict(prepared.flow_inputs, prepared.emulator_pairs).summary()
```

It has not been independently validated.

Flow, classifier and emulator training are not part of this release. Emulator training
lives in BlendEMU.

# Simulation-Based Shear Inference (SBSI)

SBSI is an integrated weak gravitational lensing (shear) calibration framework, accounting for detection and selection bias,
as well as blending.

SBSI provides one model-name-agnostic workflow with three API areas:

1. `sbs_shear.flow` — train and tune a conditional measurement flow.
2. `sbs_shear.response` — combine flow self-response and emulator blending response.
3. `sbs_shear.inference` — simulation-based shear inference (under development).

The [inference tutorial notebook](examples/sbsi_api_tutorial.ipynb) is the main
user-facing prediction walkthrough. Training and tuning use the CLI described
below.

## Image simulation and measurement

SBSI training data are generated using BlendEMU. 
BlendEMU owns rendering, measurement, and simulation-catalogue construction.
SBSI calls its supported pipeline CLI. The
[example Slurm wrapper](examples/job_blendemu.sh) runs BlendEMU steps
1 through 4b from a user-owned YAML configuration. 

## Training and tuning

Flow training is config-driven, like BlendEMU. Copy
[`examples/flow_training.yaml`](examples/flow_training.yaml), replace every
catalogue and artifact path, and run inside an appropriate compute allocation:

```bash
python -m sbs_shear flow --config my_flow.yaml --mode train
python -m sbs_shear flow --config my_flow.yaml --mode tune
```

An editable/package install provides the equivalent `sbsi` command.

`--mode train` produces the configured checkpoint and averaged `*_swaavg.pt`
checkpoint. `--mode tune` trains the explicit candidate list, evaluates each
checkpoint with the user-supplied `package.module:function` scorer on the
separate validation catalogue, and writes a ranked JSON manifest. Existing
artifacts are never overwritten. Scheduler wrappers are deployment details and
are not part of SBSI.

$R_{\rm blend}$ Emulator training and tuning remain in BlendEMU.

## Environment

SBSI is not installed in production; it runs from the checkout with `PYTHONPATH`
set. From the repository root:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

BlendEMU is needed only for the emulator step (`sbs_shear.models.load_emulator`).
Add it when you need it, and point the model roots at your own copies:

```bash
export BLENDEMU_ROOT=/path/to/blendemu
export PYTHONPATH="$BLENDEMU_ROOT:$PYTHONPATH"
```

The package can also be installed with `pip install -e .` without changing the
catalogue or model-path contract; that also provides the `sbsi` console script.

### Models

The frozen V3 artifacts — the 16-seed flow ensemble and the blending emulator —
ship in [`models/`](models/). To make `get_model("V3")` resolve against them
rather than the original cluster paths:

```bash
export SBSI_CACHE_DIR="$PWD/models"
export BLENDEMU_MODELS="$PWD/models/blendemu"
```

See [`models/README.md`](models/README.md) for the layout and checksums.

### Tests

```bash
PYTHONPATH="$PWD" python -m pytest tests/ -q
```

The suite is CPU-only, takes a few seconds, and needs no catalogue access. It must
pass on a checkout that has SBSI alone; the single BlendEMU cross-check skips when
BlendEMU is absent.

Use an interpreter that actually has pytest installed — on the LMU cluster the
`sims1` conda environment does **not**, so run the suite with the `py31`
environment instead.

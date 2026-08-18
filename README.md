# Simulation-Based Shear Inference (SBSI)

![Constgold galaxy stamps and conditional measurement-flow contours](examples/measurement_flow_contours.png)

Flow likelihood for three example galaxies. Shape measurements are also contaminated by blending, which is in practice adjusted by a separate emulator.

---

SBSI is an integrated weak gravitational lensing (shear) calibration framework, accounting for detection, selection, and blending.

SBSI provides one model-name-agnostic workflow with three API areas:

1. `sbsi.flow` — train and tune a conditional measurement flow.
2. `sbsi.response` — combine flow self-response and emulator blending response.
3. `sbsi.inference` — simulation-based shear inference (under development).

The [inference tutorial notebook](examples/sbsi_api_tutorial.ipynb) is the main
user-facing prediction walkthrough. Training and tuning use the CLI described
below.

## Installation

```bash
python -m pip install --config-settings editable_mode=compat -e /path/to/blendemu
python -m pip install -e /path/to/SBSI
```

BlendEMU is only required for emulator prediction; the flow checkpoints need SBSI and
PyTorch alone.


## Image simulation and measurement

SBSI training data are generated using BlendEMU. 
BlendEMU owns rendering, measurement, and simulation-catalogue construction.
SBSI calls its supported pipeline CLI. The
[example Slurm wrapper](examples/job_blendemu.sh) runs BlendEMU steps
1 through 4b from a user-owned YAML configuration. 

## Training and tuning

Flow training is config-driven. Copy
[`examples/flow_training.yaml`](examples/flow_training.yaml), replace
catalogue and artifact path, then:

```bash
python -m sbsi flow --config my_flow.yaml --mode train
python -m sbsi flow --config my_flow.yaml --mode tune
```

An editable/package install provides the equivalent `sbsi` command.

`--mode train` produces the configured checkpoint and averaged `*_swaavg.pt`
checkpoint. `--mode tune` trains the explicit candidate list, evaluates each
checkpoint with the user-supplied `package.module:function` scorer.

$R_{\rm blend}$ Emulator training and tuning remain in BlendEMU.



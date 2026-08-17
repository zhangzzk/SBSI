# SBSI

SBSI provides one model-name-agnostic workflow with three API areas:

1. `sbs_shear.flow` — train and tune a conditional measurement flow.
2. `sbs_shear.response` — combine flow self-response and emulator response.
3. `sbs_shear.inference` — simulation-based shear inference (under development).

Users choose every training, validation, and inference catalogue. The library
does not select data from a release name and does not require repository job
scripts. Run expensive API calls in whatever batch or cluster environment is
appropriate for your system.

The [inference tutorial notebook](examples/sbsi_api_tutorial.ipynb) is the main
user-facing prediction walkthrough. Training and tuning use the CLI described
below.

## Models are paths, not pipelines

The workflow accepts arbitrary model paths:

```python
from pathlib import Path

from sbs_shear import ModelPaths, ResponsePredictor

models = ModelPaths(
    flow_checkpoints=(Path("/models/flow_s1.pt"), Path("/models/flow_s2.pt")),
    emulator_model=Path("/models/emulator.json"),
)
predictor = ResponsePredictor.load(models, device="cuda")
```

`get_model("V3")` and `get_model("V3b")` are optional convenience presets that
return the frozen external paths. They do not change training, catalogue
selection, response estimation, or inference behavior.

## External catalogues

API boundaries accept either a pandas `DataFrame` or a user-supplied Feather,
Parquet, CSV, or pickle path. For example:

```python
from sbs_shear import (
    EmulatorPairingConfig,
    ResponsePredictor,
    get_model,
    load_catalogue,
    load_emulator,
    prepare_forward_catalogue,
    predict_blend_response,
)

models = get_model("V3")
conditions = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}
input_catalogue = load_catalogue("examples/data/example_catalog.feather")
emulator = load_emulator(models, conditions=conditions)
prepared = prepare_forward_catalogue(
    input_catalogue,
    config=EmulatorPairingConfig.from_emulator(emulator),
)
r_blend = predict_blend_response(emulator, prepared.emulator_pairs)

predictor = ResponsePredictor.load(models, device="cuda")
prediction = predictor.predict(
    prepared.flow_inputs,
    blend_response=r_blend,
)
print(prediction.summary())
```

SBSI owns the inference-time nearest-neighbour search, primary/secondary pair
table, model-training cuts, feature rescaling, V3 crowding summaries, and
alignment. `prepared.flow_inputs` is one row per retained primary;
`prepared.emulator_pairs` may contain several neighbours per primary. They are
separate views because passing the pair table to the flow would incorrectly
weight crowded primaries more heavily. BlendEMU only evaluates its trained
model on the prepared pairs.

The bundled `examples/data/example_catalog.feather` contains the truth and
orientation fields needed to build both response-model views. Measured image
catalogues remain necessary for flow training and for the future measurement-
likelihood/shear-inference API; they are not a second input to V3 response
prediction. For precomputed emulator results, users may still provide a
separate response catalogue joined on `(case, input_index)`, or an aligned
`R_blend` column. Missing emulator responses are rejected or dropped, never
replaced with zero.

## Image simulation and measurement

BlendEMU owns rendering, measurement, and simulation-catalogue construction.
SBSI calls its supported pipeline CLI instead of copying that machinery. The
[example Slurm wrapper](examples/job_generate_catalogues.sh) runs BlendEMU steps
1 through 4b from a user-owned YAML configuration. It is a deployment example, not part
of the SBSI workflow API.

## Training and tuning CLI

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

Emulator training and tuning remain in BlendEMU and will use its CLI after the
planned BlendEMU update; SBSI does not duplicate that implementation.

## Current scientific releases

[MILESTONE.md](MILESTONE.md) records V3 and V3b results and provenance. Those
names are release labels, not separate software pipelines.

## Environment

```bash
conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
python -m pytest tests/
```

The package can also be installed with `pip install -e .` without changing the
catalogue or model-path contract.

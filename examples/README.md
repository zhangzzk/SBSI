# SBSI examples

- `sbsi_api_tutorial.ipynb` is the minimal inference API walkthrough.
- `flow_training.yaml` is the explicit-path template for
  `python -m sbs_shear flow --mode train|tune`; it is configuration, not a
  scheduler script.
- `data/example_catalog.feather` is the truth-level input catalogue used by the
  BlendEMU tutorial, copied byte-for-byte from
  `blendemu/data/example_catalog.feather` (SHA-256
  `382adcac48ba7a6aab6705630c7e2e936c467cc47a5b66c6aa5d116651dbbd68`).
  It can be passed directly to SBSI's forward-catalogue and emulator-response
  APIs.
- `job_generate_catalogues.sh` is an optional Slurm deployment example. It
  calls BlendEMU's supported pipeline CLI for image simulation, shape
  measurement, and simulation-catalogue construction; none of that machinery
  is duplicated in SBSI.

The bundled truth catalogue is sufficient for both V3 response-model inputs.
`prepare_forward_catalogue` derives an object-level flow view and a pair-level
emulator view from it. Measured image catalogues produced by BlendEMU are still
needed to train the measurement flow and, eventually, to evaluate the joint
measurement likelihood for shear inference.

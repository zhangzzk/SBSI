# Native exchange type draws — bounded n22

## Fixed candidate and numerical contract

Base commit55f1220 includes the unconditional continuous-support correction.
Its complete exchange module is frozen in
`doc/evidence/scene_exchange_baseline_55f1220.py`. Only temporary target-type
draws are batched. Neighbour selection, ragged sums, relocation, and density
corrections are unchanged. Custom RNGs and non-float32/64 inputs retain their
scalar choice calls.

The candidate follows the categorical branch of
[NumPy2.2.6 Generator.choice](https://github.com/numpy/numpy/blob/v2.2.6/numpy/random/_generator.pyx):
cast to float64, accumulate each row in order, normalize by its final CDF entry,
then use a uniform and right-sided CDF search. Counting cumulative entries
less than or equal to that uniform implements this search. Batched uniform
generation must leave the same RNG state as repeated scalar calls. Existing
outer validation of the conditional probabilities remains in force.

No reduceat replacement is made for ragged sums: a different summation order
can change floating-point probabilities. This candidate deliberately tests
whether type batching alone yields a useful saving before broadening changes.

## Predeclared checks and CPU timing

Check float32/64 categorical rows with1,2,3,17,128 types, zero probabilities,
CDF boundary uniforms, and all five installed NumPy bit generators. Compare
full proposed native payloads, correction bits, RNG state, relocation, empty
scenes, single/many/ragged neighbours, support-zero marks, and changing type
probabilities. Replay mixed-block full MH trajectories with optional informed
environment payloads. AST equality pins the corrected native support helper.

The CPU benchmark uses1024 and4096 rows with empty/singleton/ordinary/many/
support-zero cases, each with and without relocation. Two warmups and11 timed
repetitions use counterbalanced arm order, identical fixed inputs, and fresh
matched seeds. The development metric is the geometric mean baseline/candidate
median proposal speed for ordinary1024/4096-row scenes without relocation;
at least1.1 is the fixed substantive-gain threshold. All cases require bitwise
payload and RNG replay. No trained model or GPU is used in this CPU stage.

A successful CPU result warrants a separately scheduled frozen trained-model
replay/cost check; it does not establish full-sampler or scientific efficiency.

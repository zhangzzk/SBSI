## cont.169 (2026-08-17) repo cleanup + blendemu decoupling (user)

Housekeeping pass over the working tree, no compute and no science change.

FIXED A BROKEN DOC CHAIN. `AGENTS.md` had been replaced by a symlink to `CLAUDE.md`, while
`CLAUDE.md` line 3 is `@AGENTS.md` -- a cycle, so the 73-line scope document (including the
blendemu boundary rules) resolved to nothing and survived only in git. Restored from HEAD and
extended with an enforced-boundary section.

DECOUPLING FROM BLENDEMU. Two new modules:
  * `sbs_shear/paths.py` -- every external root (catalogues, caches, dumps, both sim sets,
    blendemu root/models) resolved from an env var with the old value as default.
    `python -m sbs_shear.paths` prints them and flags missing ones.
  * `sbs_shear/emulator.py` -- the single place blendemu is imported, lazily, with an
    actionable ImportError. Also now the one definition of `SURVEY_CONDITIONS`, which had
    been copy-pasted as a `COND` literal into 8 scripts (and again as `RESCALE_KW` in 4 more).
Rewired 8 emulator scripts off `sys.path.insert("/home/z/.../blendemu")` +
`BLEND_MODELS = ".../blendemu/models"` onto `load_blending_predictor()`. Migrated 34 hardcoded
absolute paths in 16 scripts onto the paths module. `build_detection_measurement_catalogue.py`
(the one legitimate catalogue bridge) now goes through `import_blendemu()`. Verified the bridge
loads a real BlendingPredictor with blendemu NOT on PYTHONPATH.

Jobs: 358 job files now honour `${SBSI_ROOT}` / `${BLENDEMU_ROOT}` instead of hardcoding both
(same defaults, so behaviour is unchanged); all parse under `bash -n`. Six jobs that `cd` into
blendemu and run blendemu's own scripts (`retrain_extnbr`, `emulator_mean_residual`,
`emulator_gold_reweight`) moved to `jobs/blendemu_side/` with a README -- they are blendemu
work, not SBSI work, and their `scripts/...` lines resolve against blendemu after the `cd`.

Dead code: deleted `sbs_shear/detection_classifier.py` (a pure alias shim for a rename that had
already completed; confirmed no checkpoint pickles reference the old module path); moved
`sbs_shear/sim_stream.py` to `archive/` (used only by 6 already-archived scripts). Removed three
stale editor backups and the tracked caches; gitignored `.claude/worktrees/`.

`scripts/` is now an explicit package (`__init__.py`), and the two odd flat sibling imports
(`import infer_posterior_shape`, `import train_joint_forward`) were normalised to the
`from scripts.X import ...` style the other 6 already used, so `scripts/` no longer has to go on
`sys.path`. NOTE: scripts were deliberately NOT split into subdirectories -- 9 cross-script
imports plus ~150 job references make the churn/risk far exceed the navigational gain.

Added `README.md` (entry point + the decoupling contract) and `tests/run_tests.py`, because
`pytest` is not actually installed in `sims1` -- CLAUDE.md claimed `python -m pytest tests/`
works and it does not. Fixed dangling links to the deleted `GOALS.md`/`PIPELINE.md` and the
`Gold-v1.md` -> `Gold-V1.md` case change.

Validation: 17/17 tests pass via `python tests/run_tests.py`; every file under
`scripts/ plotting/ sbs_shear/ tests/` byte-compiles; a tokenizer audit confirms no string
literal changed except the intended path substitutions; all 22 resolved paths checked against
disk (3 absent are outputs). Known limitation: mid-pass, a regex migration corrupted 8 literals
and 5 bootstrap blocks and left 34 substitutions without their f-prefix; all were found by the
audits above and fixed, but this is why the tokenizer/resolve checks are worth keeping.

## cont.168 (2026-08-06) MATH.md sections 6-7 rewritten onto 5B (user)

Per user: recast MATH.md §6 and §7 in §5's format -- terse "Definitions" block, numbered Step
blocks each carrying one tagged equation, boxed assembly, closing assumptions table -- and point
them at INFERENCE.md §5B (5.3) instead of §5C (5.8).

§6 "The Eulerian form: derivation of INFERENCE.md (5.3)". Per a follow-up request it is now
STANDALONE -- it makes no reference to §5 and borrows none of §5's equations (M.1-M.11), verified by
regex over the section's line range. That cost three additions: Step 4 writes Louis's two lines out
instead of citing (M.10); Step 5 derives its own selection split (log p_keep = log W + log A - log P
=> both moments centred) and its own Newton step, instead of assembling into §5's (M.8); and the
§5-vs-§6 comparison table and "Contrast with §5" line were deleted. Eight steps: (1) model with
gamma on the prior (M.12); (2) generator u = -(v.grad log p0 + div v) (M.13) from the continuity
equation, with the composition-is-not-pushforward warning kept; (3) s_i = E_w[u] by Fisher (M.14);
(4) I_i by Louis (M.15); (5) selection + Newton step; (6) population pair (M.16) = (5.3b);
(7) assemble (M.17) = (5.3); (8) (M.18), the change of variables, now phrased WITHOUT naming §5 --
it just states that the same A results from shearing the samples with p0 fixed, which is a fact
about (M.12) itself. Also added a self-contained "why it is unbiased at first order" paragraph and a
"Regularity" line (differentiation under the integral twice; Newton step exact to O(gamma^2),
iterate to remove; common random numbers inside P_pass). NEW vs the old text: a
"where R_blend enters" note (u carries the spin-2 neighbour components, so s_i picks it up with no
separate term, and it dies by isotropy exactly when L_k is blind to the neighbour position angle),
and an E1-E5 requirements table imported from INFERENCE.md §5B.3 (grad log p0 over the whole scene,
scene prior/clustering, PSF+depth conditioning, P_det and its sheared-prior integral, ESS-not-node-
count with the ratio-estimator bias). Isotropy result (<s>_sel = 0 exactly, I_sel = iota*1) kept.

§7 "Relation to INFERENCE.md (5.3)". Substantive change, not just relabelling: against 5B the
Bartlett denominator is NOT cheaper. Louis's extra ingredient d_gamma u_k is per NODE and amortized
across the catalogue (O(N_node) on an O(N_gal*N_node) estimator), whereas on the Lagrangian route
phi'' is per (object,node) and costs ~2-3x. So (M.6)'s value against (5.3) is as the free internal
consistency check of §5B.2, not as a cheaper estimator. (M.19) drift formula retained, retagged
m_(M.6) - m_(5.3), with a pointer that §8's tables label the same denominator (5.8).

Also fixed the doc intro: §7 no longer "compares the two" 5C denominators, and the pointer to the
equivalence proof now says Step 8 (it was Step 6 before this session). Equation tags M.1..M.19
unchanged and still unique, so §8/§9 cross-references hold; §7's back-reference was repointed from
"§6 Step 7" to (M.18). §6 is 190 lines (was 137 before this session): prose is tighter throughout, but
standalone-ness costs the re-derived Louis/Newton/selection steps and the E-table. Docs only, no
compute.

## cont.167 (2026-08-06) preliminary slide deck (user)

Built `slides/sbsi_slides.tex` -> `slides/sbsi_slides.pdf`, 8 beamer slides (16:9, minimal),
per user request: (1) overview; (2) core INFERENCE.md 5B posterior estimator (score u, per-object
w_k prop p_flow*P_det, s_i/I_i, population Pi = P_pass*P_det, boxed ghat = (sum s_i - N<s>_sel)/
(sum I_i - N I_sel), plus the cut-cancels / detection-does-not rule and latent neighbours);
(3) trained-models table (V2 dom6x6 flow arch+training read from the s501 swaavg checkpoint
metadata; response-aware detection MLP lam300 from det_response_mlp_lam300_s7.pt; BlendEMU
in-domain tuned lookup); (4-8) all five fiducial figures, one slide each except fid_fig1+fid_fig3
which share the training/seed-spread slide.
Second pass per user: the math was split in two -- slide 2 "fundamentals" (log evidence, score,
Fisher's identity + generator u, Bartlett E0[s]=0 and I=Var0[s], linear expansion E_gamma[s]=I*gamma
-> ghat=sum s_i/sum I_i) and slide 3 "estimator, selection, R_blend" (node bank, w_k/Pi_k, the boxed
selection-corrected ghat, the cut-cancels rule, and s=s_self+s_nbr -> R=R_self+R_blend by covariance
with the spin-2 non-degeneracy caveat). The models table and its two figures were also split. Numbers quoted from figure annotations and
`results/constgold_neardomain_table.npz` (worktree selbias-plot): R_sim=0.8605, R_flow=0.7258,
R_blend=0.1358, no-cut m=-0.12+-0.15% (16 seeds, N=11,674,408), R>0.70" cut m=+4.46+-0.14%.
Third pass per user, after the MATH.md §6 rewrite (cont.168): the two math slides were remade to
follow MATH.md §6 rather than INFERENCE.md §2/§5B, i.e. the standalone Eulerian derivation.
Slide 2 is now "The model, and its two derivatives" (§6 steps 1-4): A(x_hat|gamma) = int L*P_det*p_gamma
with p_gamma = S_gamma # p_0; the generator u = -(v.grad log p_0 + div v) from the continuity equation;
then Fisher s_i = E_w[u] and Louis I_i = -E_w[d_gamma u] - Var_w(u) side by side as the first and second
derivatives of log A. Slide 3 is "Selection, the estimator, and R_blend" (§6 steps 5-7): the
log p_keep = log W + log A - log P decomposition, the point that W being gamma-free centres BOTH moments
by the same two population numbers, the Pi-weighted pair, and the boxed estimator now shown as the
Newton step ghat = -l'(0)/l''(0) = (sum s_i - N<s>_sel)/(sum I_i - N I_sel). Net changes vs the
second pass: Bartlett (I = Var_0[s]) and the linear-expansion route are GONE from slide 2, replaced by
Louis's curvature (which is what the pipeline computes); I_i moved from slide 3 to slide 2; the
selection derivation is now shown rather than asserted; the R_blend framing changed from the additive
s = s_self + s_nbr => R = R_self + R_blend covariance split to §6's version -- u carries the spin-2
neighbour components so s_i picks the blend response up with no separate term, vanishing only when the
flow is blind to the neighbour position angle. Slides 1 and 4-8 untouched.
Build: `cd slides && pdflatex sbsi_slides.tex` (graphicspath ../figures/). 8 pages, verified by
rendering every changed page to PNG; residual overfull vbox 2.99pt (slide 3) and 9.22pt (two figure
slides), all confirmed non-clipping. Docs only, no compute.

## cont.166 (2026-07-27) beta-by-size plot refinements (user)

Per user: (1) fixed true-Re bins [0,0.2,0.5,0.8,1.2] (was quantile) via new --size-edges arg in
scripts/eval_detection_beta_bysize.py; (2) removed all titles; (3) removed legend titles + larger font;
(4) x-labels -> just "blendedness beta" (dropped parenthetical + "(iso)" tick). Re-ran sim (job 15288728
-> detection_beta_bysize_v3.npz) + model (job 15288888 -> model_beta_bysize_v3.npz);
plotting/plot_detection_beta_bysize.py -> figures/fig_detection_beta_bysize.png. NOTE: top size bin
Re[0.8,1.2) is sparse in the fixed faint band 25.5-26.5 (big faint galaxies rare) -> a couple beta cells
dropped, large error bars, sim much deeper than classifier there (classifier under-predicts det-bias for
large faint galaxies; sim iso -16.7% -> -20.5%, model ~-3 to -6%). Branch ablation-v1-to-v2.

## cont.165 (2026-07-27) beta-by-size det-response: SIM vs MODEL, two facets (size + mag)

Reworked cont.164's beta-by-size plot per user: (1) more Re bins, (2) beta computed for ALL (isolated
-> beta=0, leftmost x), (3) left panel only, (4) overlay classifier MODEL vs constgold SIM, plus a 2nd
panel faceted by true MAG instead of size. scripts/eval_detection_beta_bysize.py rewritten to emit BOTH
facets (SIZE__/MAG__ keys) in one pass (job 15288392 -> detection_beta_bysize_v2.npz): SIZE facet 5 Re
bins x 6 beta bins at fixed mag 25.5-26.5; MAG facet 4 r-mag bins (24.5-26.5) pooling size. New
scripts/eval_model_beta_bysize.py runs the cont.160 response-regularized classifier (det_response_mlp_
lam300_s7.pt) on the det_meas parent, forming its induced detection response by the same centered-finite-
difference-through-the-shear-map estimator, binned on the SAME (primary x beta) edges (job 15288413 ->
model_beta_bysize_v2.npz). plotting/plot_detection_beta_bysize.py -> figures/fig_detection_beta_bysize.png
(2 panels, solid+filled=sim, dashed+open=model).
FINDING: sim and model track well in both facets -- detection response goes more negative as beta grows
and as size/faintness grows; ordering + shape match. Model slightly OVER-predicts the bias at the large-
size / faint tail (e.g. SIZE Re[0.41,9.83): sim -6.0..-10.5% vs model -4.7..-5.5%; the sim's deepest
high-beta point -10.5% is not reached by the model). Confirms the classifier captures the qualitative
severity dependence; residual = large/faint tail. Files: scripts/eval_{detection,model}_beta_bysize.py,
jobs/job_{detection_beta_bysize,model_beta_bysize}.sh, plotting/plot_detection_beta_bysize.py. Branch
ablation-v1-to-v2.

## cont.164 (2026-07-27) nbr_flux_near mag-curves + does size-binning rescue beta?

Two follow-ups to cont.163. (1) User plot: det-bias vs nbr_flux_near in 4 true-mag bins (24.5-26.5),
isolated=nbr_flux_near==0. scripts/eval_detection_nbrflux.py --mag-edges (job 15288074), npz
detection_nbrflux_magbins_v1.npz, plotting/plot_detection_nbrflux_magbins.py ->
figures/fig_detection_nbrflux_magbins.png. Each mag bin: det-bias deepens MONOTONICALLY with
nbr_flux_near, isolated is the least-biased anchor, <Re> flat along each curve (no size confound).
Faint 26.0-26.5 most biased (iso -2.15% -> -4.0%); brighter bins iso -0.4..-0.6% -> -1.7..-2.8%.

(2) "What if we keep beta but bin by size?" scripts/eval_detection_beta_bysize.py (job 15288102, mag
band 25.5-26.5, 3 Re bins x 6 beta bins), npz detection_beta_bysize_v1.npz,
plotting/plot_detection_beta_bysize.py -> figures/fig_detection_beta_bysize.png. ANSWER: PARTIALLY.
- Small Re[0.01,0.20] <Re>0.13: iso -0.02%, beta 0->-0.36%, <Re> FLAT -> clean.
- Medium Re[0.20,0.34] <Re>0.27: iso -0.29%, beta ->-1.72%, <Re> FLAT -> clean (blended>=iso).
- Large Re[0.34,9.83] <Re>0.49: iso -4.05% but low-beta -1.59% (<iso, artifact BACK), ->-9.4%; and
  <Re> CLIMBS 0.41->0.55 along beta -> beta still sorts on size inside this wide (equal-count) tertile.
So size-binning rescues beta only where the size bin is narrow; beta's size entanglement is continuous,
and the large-Re tail keeps tracking size unless binned finely (or a 2D beta x size grid). nbr_flux_near
sidesteps it by construction (no primary-size term) -> recommended severity axis; beta as cross-check.
Files: scripts/eval_detection_{nbrflux(+--mag-edges),beta_bysize}.py, jobs/job_detection_{nbrflux_magbins,
beta_bysize}.sh, plotting/plot_detection_{nbrflux_magbins,beta_bysize}.py. Branch ablation-v1-to-v2.

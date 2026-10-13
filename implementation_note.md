# Two-component GPE implementation decisions

## Scope and modification boundary

Only `two_component_GPE` is a development target. The sibling `GPE_simulator`
(single-component package) is **read-only reference material**. Never modify it,
its Git history, notebooks, environments, configurations or results. This rule
also applies to notebook bootstrap cells and future maintenance.

Keep the reference layout and Experiment → Launch → Monitor GUI logic. The
reviewed eight physics files live in `gpe/core/`; the import package remains
`gpe` to minimize changes to commands and portable run bundles.

Maintain two Git branches: `ssh` for the existing SSH/OpenSSH/Slurm workflow,
and `colab` for a separate local GPU launch workflow with persistent Drive output.
Shared physics/configuration/I/O changes should be ported between branches;
launch-specific changes should remain on their intended branch.

## Physical model and units

Two density-coupled hyperfine components in the F=1 manifold of 39K; no coherent
RF/Raman coupling. The exact magnetic field and scattering lengths are inputs,
not inferred from the state labels. Example values are illustrative, not a
calibration for a particular field.

Use l=1 micrometre, E0=hbar^2/(2 m l^2), and t0=2 m l^2/hbar, as in the reference.
The dimensionless kinetic operator is -Laplacian. Each initial Gaussian is
individually normalized to integral |psi_i|^2 dV=1.

- beta11 = 8 pi N1 a11/l
- beta12 = 8 pi N2 a12/l
- beta21 = 8 pi N1 a12/l
- beta22 = 8 pi N2 a22/l

YAML scattering lengths a11, a22, a12 are in Bohr radii; `get_beta` converts them.
N1 and N2 are fixed reference populations. Physical densities are
N_i |psi_i|^2/l^3. Real-time losses reduce the field norms; do not renormalize the
fields or recompute beta from the remaining populations. Continued stages keep
both reference populations. Scattering lengths may change between stages.

The common box, absorber and static disorder act on both components. Each has
an additional independent static and time-dependent potential. Compute both
local densities before overwriting either field. Both dynamic half-steps use
the stage midpoint time. Stage time resets to zero at each stage.

## Accepted numerical choices and open physics questions

The reviewed real-time conservative implementation passed small CPU mathematical
checks: analytic coupled plane waves, separate norm conservation, reversibility,
and approximately fourfold error reduction when halving dt. This is evidence
for the tested equations and second-order temporal convergence, not a certificate
of CUDA execution, spatial convergence or experimental model validity.

**Imaginary time is retained deliberately.** Its nonlinear half-step freezes the
density in an exponential, followed by separate normalization after the full
step. Measured transient errors and stationary residuals decrease approximately
linearly with dt. A tiny step-to-step difference can conceal a timestep-biased
stationary state. The user accepts this implementation for now; discuss a change
before replacing its numerical method.

The target is relaxation inside the finite box during an intermediate imaginary
time, before appreciable leakage. It is not the global ground state over the
whole periodic domain. Compare preparations at the same total imaginary time
with smaller dt. Seek a window where both interior densities, interface/relative
density and stationary residual have settled while exterior populations remain
small. Multiple initial profiles may be needed to investigate competing states.

The absorber's local nonlinear phase also freezes density during damping. The
user accepts this boundary approximation to suppress reflected/returning flux;
check sensitivity of interior observables to absorber width and strength.

Unchanged choices that merit discussion for research use:

- The box uses maximum(tube, sheet), not a sum; additive optical beams have a
  different barrier where they overlap.
- Per-grid-cell random potential noise changes its physical correlation scale
  under grid refinement. Hold the intended physical disorder fixed in comparisons.
- The kinetic cutoff removes modes and population. The final nonlinear half-step
  can regenerate high-k content; the cutoff does not establish alias-free evolution.
- Normalization of a zero/nonfinite field currently has no explicit core error
  guard. Monitor norms and finiteness; do not interpret normalization as convergence.

No physics formulas were changed while packaging these files.

## FFT and memory

One explicit 3D C2C plan is reused sequentially for both components and directions.
`overwrite_x=True` permits in-place transformation. Every returned pointer is
checked against that transform's input, raising `InPlaceFFTError(RuntimeError)`
on a mismatch. Backend OOM/cuFFT exceptions retain their original cause. A failed
step may be partially applied; load a complete saved pair to start a new run.

At 1024^3, each complex128 field occupies 16 GiB. Persistent real-time arrays
(two fields plus the complex static potential) occupy 48 GiB; imaginary time
uses 40 GiB with a real potential. These are NOT peak memory estimates: FFT
workspace, potential construction, memory pools and diagnostics add allocations.
Diagnostics process the components sequentially but need an additional Fourier
field and other working storage. A fit on an 80 GB A100 has not been established.
FFT dominates the user's benchmark; explicit planning is not itself a guaranteed
speedup because CuPy already caches plans.

## Configuration, continuation and outputs

Use schema_version: 3. Reject single-component schema 2 instead of guessing a
second component. `cloud` contains N1, N2, a11, a22, a12. `potential1` and
`potential2` each contain static/dynamic enabled flags and expressions.
`initial_state.component1` and `initial_state.component2` each contain type
(gaussian/load/continue), sigma and path. The GUI exposes separate choices and
paths. Existing schema-3 pair-level type/sigma1/sigma2/path settings are converted
on import; a legacy JSON path selects its corresponding component for each field.
New exports use the component form. Older runners cannot read this extension.

A saved pair consists of psi1_STEP.npy, psi2_STEP.npy, grid.npz and psi_STEP.json.
The manifest is published last, after both slab-written fields are flushed and
closed. A .partial file or fields without the manifest are not a completed pair.
Independent loading takes a psi1_STEP.npy or psi2_STEP.npy path and automatically
reads psi_STEP.json and grid.npz beside it. Keep these original names. The other
NPY member is not read or required. Requiring the completed manifest retains
grid/reference-population checks and prevents reuse of unfinished snapshots.
Arbitrary NPY files without this metadata are not supported by this loader.

The two inputs can come from different runs or steps, or one can be Gaussian
or continued. This constructs a new initial condition, not necessarily a state
that has evolved together. A source component can initialize either destination;
the destination's reference N must equal that source component's saved reference
N. Loading preserves complex amplitudes, phases and norms without normalization.
The same file may initialize both components; they receive distinct GPU arrays.
All selected files are checked before new fields are allocated. Transfers stay
slab-wise, and the runner releases replaced components before loading their
replacements. Continuation checks reference N only for the continued component.
This starts a new stage; it is not an automatic restart of its clock.

Each component has its own monitor subfolder. Scalar `norm`, `N_atoms`, `E_kin`
and `Lz` retain the reference definitions for that component; E_kin is kinetic
energy, not the total interacting energy. Shared cadence settings apply to both.
NPZ monitor samples are published after writing completes; an interrupted final
CSV row is ignored by the viewer. Numerical output is excluded from Git.

## Launch and persistence

SSH keeps the reference portable bundle, environment/module selection, SSH
authentication, upload, Slurm submission/cancellation and small-file monitor
refresh. Full wavefunctions stay on the remote machine unless explicitly copied.

Colab uses a user-selected Drive simulations parent (default
`/content/drive/MyDrive/simulations`). Each new launch name is a new folder;
existing names are rejected. It contains config/run.yaml, notes.txt, data/,
monitor/, logs/, run.json and the exact copied runtime sources. Prepared output
roots are fixed inside that folder; stage overrides cannot escape it. The GUI
reopens monitor files directly from the same Drive folder after reconnection.

Mounted Drive is the write destination throughout the run, not a final export
step. A browser disconnect is different from deletion/termination of the runtime:
computation cannot continue after the GPU runtime is destroyed. Unsaved steps and
an in-flight write can be lost; periodic snapshot and diagnostic cadences still
matter. Drive writes depend on its runtime-mounted filesystem and can fail due
to disconnection/quota. Flush/close/rename does not prove remote cloud replication
has completed. Verify a completed sample in Drive before a long experiment.
See https://research.google.com/colaboratory/faq.html and the Colab I/O notebook.

## Validation boundary

CPU tests cover schema, paired continuation/snapshots, output scheduling, GUI
HTTP/session behavior, bundles and mocked launch controls. Actual GPU tests in
`tests/test_propagators.py` exercise coupled phases, each driven component,
cutoff branches, normalization and FFT pointer reuse. Run them on the target
GPU before production simulations. They skip when CUDA is unavailable.

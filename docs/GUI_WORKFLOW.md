# Experiment → Launch → Monitor

The interface follows the single-component package, with two populations, three
scattering lengths, two Gaussian widths and per-component static/dynamic potentials.
Each stage still has one grid, clock, shared trap, absorber and disorder field.
The shared monitor/save intervals apply independently to both components.

Load schema-3 YAML or build an experiment in the editor. Formula previews support
the same validated arithmetic as the runner. Browser drafts use a separate key
from the single-component GUI. Preparing a run captures the current draft, run
name, notes and runtime sources. Editing the draft does not change a prepared run.

Use the `ssh` branch for cluster controls and `colab` for Drive-backed Colab runs.
See that branch's README and notebook. The monitor stage menu includes component
subfolders, so density, phase, spectra and scalar history stay component-resolved.
Under Evolution, each component has its own initial-state choice: Gaussian,
Load a saved wavefunction, or Continue previous stage. For a load, enter the
absolute target path to the selected psi1_STEP.npy or psi2_STEP.npy. Its saved
psi_STEP.json metadata and grid.npz must remain alongside it; no second path
is needed for the metadata. The other component can use another file or source
type. Both fields must match the simulation grid, and each destination reference
atom number must match its selected source's reference population.

Saved snapshots still contain both fields. Legacy pair-load YAMLs remain
readable and expand into two component choices. Loading preserves each field's
norm and phase; the new stage clock starts at zero. Prepare a new named run after
changing these settings so the launch uses the new configuration.

The only editable software target is this repository. GPE_simulator is read-only.

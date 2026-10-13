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
The full wavefunctions are stored as paired snapshots; load their JSON manifest.

The only editable software target is this repository. GPE_simulator is read-only.

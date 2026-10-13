# Two-component experiment editor

Run `python -m gui_periphery.app` after installing `gui_periphery/requirements.txt`.
The CPU-only editor shares schema-3 validation with the runner. It keeps the
Experiment → Launch → Monitor layout; each component has separate static/dynamic
expressions and the cloud has N1, N2, a11, a22, a12. Scattering lengths are in Bohr radii.

See the repository README for the selected branch's launch workflow and
`implementation_note.md` for physics choices. The reference GPE_simulator is read-only.

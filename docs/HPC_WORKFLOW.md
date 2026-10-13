# SSH / Slurm workflow

Use the `ssh` branch. The local GUI preserves the reference package's connection
sharing, password/key/MFA prompts, preparation, upload, environment setup,
submission, scheduler refresh and cancellation.

1. Run `python -m gui_periphery.app` (or `Open GPE.command` on macOS).
2. Configure both components in Experiment.
3. In Launch select CSD3, name the run, enter notes, GPU account, partition,
   wall time, Python/module settings and the matching CuPy wheel.
4. Prepare, connect, upload to an absolute remote parent directory, set up the
   environment if needed, and submit. A prepared bundle captures the exact YAML
   and runtime source hashes. Later editor changes need a new prepared run.
5. Monitor refresh transfers diagnostic files and log tails, not wavefunctions.
   Select a stage/component to inspect its saved diagnostics.

SSH authentication uses the existing OpenSSH helper. Credentials are not written
to the launch profile. Existing run folders are never overwritten. The remote
scheduler, CUDA environment and project account must be supplied by the user.

For loading a previous state, supply an absolute target path to its paired
`psi_STEP.json` manifest; both component NPY files and grid.npz must be beside it.
No external snapshots are uploaded automatically.

This two-component implementation has local/mocked regression coverage; it has
not yet been submitted to a real cluster. Never alter the sibling GPE_simulator.

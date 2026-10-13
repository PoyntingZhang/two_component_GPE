# Two-component GPE simulator — Colab branch

This branch runs the two-component solver in a Colab GPU runtime and saves
scheduled output **directly to Google Drive**. Use `git switch ssh` for the
separate SSH/Slurm implementation. Commit edits before switching branches.
The sibling `GPE_simulator` is read-only reference material.

## Start

Open [notebooks/Colab.ipynb](notebooks/Colab.ipynb) in Colab. Select a GPU runtime,
mount Drive, and set the repository path and simulations parent. The default is:

```python
from gui_periphery.colab import show
show(output_parent='/content/drive/MyDrive/simulations')
```

The notebook installs GUI dependencies and opens Experiment → Launch → Monitor.
Configure both components, enter a launch name and purpose, prepare on Drive,
then launch. Use the CuPy installation button only if the runtime needs it.
Choose the wheel matching the runtime's CUDA environment.

```text
simulations/
  mixture_trial/
    config/run.yaml       Validated configuration actually used
    notes.txt             Brief purpose and description
    data/<run>/<stage>/
      psi1_00001000.npy
      psi2_00001000.npy
      psi_00001000.json    Complete snapshot-pair manifest
      grid.npz
    monitor/<run>/<stage>/
      component_1/        Scalars, spectra and selected images
      component_2/
    logs/
    run.json              Creation information and source/config hashes
    gpe/                  Exact runtime source copy
    config/*.py
    run.sh
    environment.sh
```

Existing launch names are never overwritten. Colab preparation rejects output
outside mounted Drive. The prepared YAML routes output inside that launch folder;
stage overrides cannot escape it. Completed pairs and NPZ samples are published
only after writing completes. Incomplete `.partial` files are not snapshots.

A terminated runtime cannot continue computing, and unsaved steps/in-flight
writes can be lost. Choose periodic saves; a final-only snapshot cannot protect
an interrupted run. The mounted filesystem's flush/rename is not proof of cloud
replication. Check a completed output in Drive before a long run. See
[Colab's Drive guidance](https://research.google.com/colaboratory/faq.html).

After reconnecting, rerun the setup cells and use **Reopen a saved run** with its
Drive launch folder. Monitoring reads those files directly. To continue evolution,
prepare a new run and choose **Load a saved wavefunction** independently for each
component under **Experiment → Evolution**. Enter the full path to its
`psi1_STEP.npy` or `psi2_STEP.npy`. Keep its `psi_STEP.json` metadata and `grid.npz`
beside it; the loader finds them automatically. The fields can come from different
runs, or one can be a Gaussian or continue the previous stage. Match the grid and
each selected source's reference atom number. See
[the independent-load example](config/examples/load_independent.yaml).
Existing pair-load YAMLs remain supported. Loading preserves norms and starts a
new stage clock, not an automatic exact-time restart.

## Model and validation

The reviewed physics is in `gpe/core/`, unchanged by packaging. Schema 3 adds N1,
N2, a11, a22, a12, separate widths and `potential1`/`potential2`. Example scattering
lengths are illustrative, not a calibrated 39K magnetic-field prescription.
Imaginary-time preparation retains its accepted first-order timestep bias.
Read [implementation_note.md](implementation_note.md) for units, limitations,
memory considerations and the modification boundary.

```sh
python -m gpe.main config/examples/kick_and_hold.yaml --check
python -m pytest tests -q
python -m unittest discover -s tests -p test_propagators.py -v  # actual GPU checks
```

Local tests simulate Drive mounting and GPU launch controls. A real Colab/GPU
launch, cloud persistence across runtime destruction, and 1024³ A100 memory use
still require target-machine verification.

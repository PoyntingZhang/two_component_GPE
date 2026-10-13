# Two-component GPE simulator — SSH branch

Density-coupled two-component 39K solver with the reference package's
Experiment → Launch → Monitor interface. The sibling `GPE_simulator` is read-only.
Important physics and storage decisions are in [implementation_note.md](implementation_note.md).

Use `git switch ssh` for SSH/Slurm launches and `git switch colab` for the dedicated
Colab workflow. Commit edits before switching branches. No remote repository is
configured automatically.

## SSH workflow

Install `gui_periphery/requirements.txt`, then run:

```sh
python -m gui_periphery.app
```

On macOS, `Open GPE.command` creates the local GUI environment and opens it.
Configure both populations, all three scattering lengths, and each component's
potentials. In Launch, choose a run name and description, enter the cluster
settings, prepare, connect, upload and submit. Monitor's stage selector includes
`component_1` and `component_2`; full wavefunctions remain remote.

```sh
python -m gpe.main config/examples/kick_and_hold.yaml --check
python -m gui_periphery.prepare_run config/examples/kick_and_hold.yaml \
  --name mixture_trial --notes 'Test opposite component kicks'
```

Schema 3 distinguishes this model from the single-component package. Example
scattering lengths are illustrative; choose your experimental values. A loaded
state must name a `psi_XXXXXXXX.json` paired snapshot, not an individual NPY.

## Layout and checks

`gpe/core/` contains the reviewed physics unchanged. `stage.py`, `stage_factory.py`
and `launch.py` carry both fields; `gpe/io/` saves pairs and component diagnostics.
`config/` remains CPU-only. `gui_periphery/` preserves the editor and SSH controls.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
# On the target CUDA GPU, after installing the matching CuPy build:
python -m unittest discover -s tests -p test_propagators.py -v
```

CPU tests cannot establish A100 memory capacity or CUDA numerical correctness.
The accepted imaginary-time method has first-order timestep bias; see the notes
before interpreting its result as a stationary state.

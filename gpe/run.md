# Runner

From the repository or a prepared runtime bundle:

```sh
python -m gpe.main config/run.yaml --check
python -u -m gpe.main config/run.yaml
```

The direct example is `config/examples/kick_and_hold.yaml`. The runner requires
NumPy, PyYAML and one CuPy wheel appropriate to the installed CUDA environment.

Schema 3 stages evolve psi1 and psi2 together. Use `initial_state.type: continue`
to carry both fields and their remaining norms between stages. Use `load` with
a complete paired `psi_STEP.json` snapshot manifest to initialize a new stage.
Stage time always starts at zero. Intervals count completed steps; zero disables.

Output retains timestamp/stage subdirectories. Snapshots contain both fields.
Monitor subdirectories `component_1` and `component_2` use the original scalar,
spectrum, density, TOF and phase conventions. `E_kin` is kinetic energy only.

For timing with all output disabled:

```sh
python -m gpe.benchmark --grids 64 128 --steps 20 --warmup 3 --repeats 3
```

Read [implementation decisions](../implementation_note.md) for normalization,
FFT memory, imaginary-time limitations and the read-only reference boundary.

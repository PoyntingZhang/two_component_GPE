from .io.save import save_outputs


def launch_stage(stage):
    """Run a prepared stage and return its final GPU wavefunction pair."""
    save_outputs(stage, 0, initial=True)
    dynamic = (stage.potential1_params["dynamic"]["activate"] or
               stage.potential2_params["dynamic"]["activate"])
    for i in range(stage.steps):
        if dynamic:
            stage.psi1, stage.psi2 = stage.propagator(stage, i * stage.dt)
        else:
            stage.psi1, stage.psi2 = stage.propagator(stage)
        save_outputs(stage, i + 1, final=(i + 1 == stage.steps))
    return stage.psi1, stage.psi2

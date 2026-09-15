from worker import core


def measurements(log, tasks, policy, *, exit_code, core_executable="invar"):
    return core.invoke(["inspect", "measurements", "--log", log, "--tasks", tasks,
                        "--policy", policy, "--exit-code", exit_code], executable=core_executable)


def performance(manifest, tasks, policy, *, core_executable="invar"):
    return core.invoke(["performance", "--manifest", manifest, "--tasks", tasks, "--policy", policy],
                       executable=core_executable)

from worker.baseline import data as product_data


def read(directory, *, services, executable):
    return services.invoke(["policy", "inspect", "--checkpoint", directory], executable=executable)


def check_selection(selected, settings):
    expected = {name: selected[name] for name in ("adapter", "tokenizer", "base", "assembly")}
    if product_data.identities(settings, behavior=True) != expected:
        raise ValueError("Native inference settings differ from the selected policy description")


def check_loaded(selected, backend):
    if tuple(backend.identity) != (selected["model"], selected["revision"]):
        raise ValueError("Loaded model revision differs from the selected policy description")
    check_selection(selected, backend.settings)


def successor(directory, expected, *, services, executable):
    selected = read(directory, services=services, executable=executable)
    if selected != expected:
        raise ValueError("Published policy description differs from the retained inference selection")
    return selected

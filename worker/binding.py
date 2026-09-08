def trainable_names(model):
    return {id(value): name for name, value in model.named_parameters() if value.requires_grad}


def parameters(model, optimizer):
    named = trainable_names(model)
    groups = optimizer.param_groups
    actual = [id(value) for group in groups for value in group["params"]]
    if not named or len(actual) != len(set(actual)) or set(actual) != set(named):
        raise RuntimeError("Optimizer parameter binding differs from the trainable model")
    saved = optimizer.state_dict()["param_groups"]
    identities = [key for group in saved for key in group["params"]]
    return dict(zip(identities, (named[key] for key in actual), strict=True))


def require(condition):
    if not condition:
        raise RuntimeError("Checkpoint parameter binding differs from the consumer optimizer")


def matches(value, expected):
    if not isinstance(value, dict):
        return False
    return (all(type(key) is int and key >= 0 and isinstance(name, str) for key, name in value.items())
            and len(value) == len(expected) and set(value.values()) == set(expected))


def group_ids(declared):
    require(all(isinstance(group, dict) and isinstance(group.get("params"), list) for group in declared))
    identities = [key for group in declared for key in group["params"]]
    require(all(type(key) is int for key in identities))
    return identities


def groups(source, target):
    declared, names = source
    actual, expected = target
    require(isinstance(declared, list) and len(declared) == len(actual))
    identities = group_ids(declared)
    require(len(identities) == len(set(identities)) and set(identities) == set(names))
    for before, after in zip(declared, actual, strict=True):
        require({names[key] for key in before["params"]} == {expected[key] for key in after["params"]})


def restore(training, model, optimizer):
    expected = parameters(model, optimizer)
    names = training.get("parameters")
    require(matches(names, expected.values()))
    saved = training["optimizer"]
    actual = optimizer.state_dict()["param_groups"]
    groups((saved["param_groups"], names), (actual, expected))
    indices = {name: key for key, name in expected.items()}
    slots = saved["state"]
    require(isinstance(slots, dict) and all(type(key) is int and key in names for key in slots))
    state = {indices[names[key]]: value for key, value in slots.items()}
    ordered = [{**before, "params": after["params"]} for before, after in zip(saved["param_groups"], actual, strict=True)]
    return {**training, "parameters": expected, "optimizer": {**saved, "state": state, "param_groups": ordered}}


def observation(training):
    names = training["parameters"]
    optimizer = training["optimizer"]
    state = {names[key]: value for key, value in optimizer["state"].items()}
    ordered = [{**group, "params": sorted(names[key] for key in group["params"])} for group in optimizer["param_groups"]]
    return {**training, "parameters": sorted(names.values()),
            "optimizer": {**optimizer, "state": state, "param_groups": ordered}}


def schema(state):
    result = {}
    for name, tensor in state.items():
        if not name.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"Unsupported adapter parameter in the default LoRA profile: {name}")
        parameter = name.removesuffix(".weight") + ".default.weight"
        result[parameter] = tensor
    return result

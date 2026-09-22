def response_history(actual, expected, *, asynchronous):
    if len(actual) != len(expected):
        raise ValueError("Native cached response context differs from the owned observations")
    # The native async runner stores -1 before the sampled token reaches CPU.
    # Actual GPU model inputs are checked separately; this does not fill in or
    # claim to observe an unresolved CPU history word.
    for recorded, observed in zip(actual, expected, strict=True):
        if recorded != observed and not (asynchronous and recorded == -1):
            raise ValueError("Native cached response context differs from the owned observations")

import hashlib

FORMAT = 'invar-inference-materialization-v1'
FIELDS = ('adapter', 'tokenizer', 'base', 'assembly')
LEARNING_FORMAT = 'invar-learning-materialization-v1'
LEARNING_FIELDS = ('policy', 'learner', 'tokenizer', 'base', 'assembly', 'reference')


def image(identities):
    encoded = '\0'.join((FORMAT, *(identities[name] for name in FIELDS))).encode('ascii')
    return {'artifact': hashlib.sha256(encoded).hexdigest(), 'profile': identities['assembly']}


def invocation(value):
    return {'binding': value.binding(), 'program': value.program}


def learning(loaded):
    encoded = '\0'.join((LEARNING_FORMAT, *(loaded[name] for name in LEARNING_FIELDS))).encode('ascii')
    return {'artifact': hashlib.sha256(encoded).hexdigest(), 'profile': loaded['assembly']}

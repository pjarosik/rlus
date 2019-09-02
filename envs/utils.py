import copy
import jsonpickle

def copy_and_apply(src, deep=False, **kwargs):
    if deep:
        cpy = copy.deepcopy(src)
    else:
        cpy = copy.copy(src)
    for k, v in kwargs.items():
        setattr(cpy, k, v)
    return cpy


def to_string(obj):
    return jsonpickle.encode(obj)

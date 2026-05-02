from collections import abc
from collections.abc import Generator, Iterable


def is_array(x):
    "`True` if `x` supports `__array__` or `iloc`"
    return hasattr(x, "__array__") or hasattr(x, "iloc")


def is_iter(o):
    "Test whether `o` can be used in a `for` loop"
    # Rank 0 tensors in PyTorch are not really iterable.
    return isinstance(o, (Iterable, Generator)) and getattr(o, "ndim", 1)


def is_coll(o):
    "Test whether `o` is a collection (i.e. has a usable `len`)"
    # Rank 0 tensors in PyTorch do not have working `len`.
    return hasattr(o, "__len__") and getattr(o, "ndim", 1)


def listify(o=None, *rest, use_list=False, match=None):
    "Convert `o` to a `list`"
    if rest: o = (o,) + rest
    if use_list: res = list(o)
    elif o is None: res = []
    elif isinstance(o, list): res = o
    elif isinstance(o, str) or isinstance(o, bytes) or is_array(o) or isinstance(o, abc.Mapping): res = [o]
    elif is_iter(o): res = list(o)
    else: res = [o]
    if match is not None:
        if is_coll(match): match = len(match)
        if len(res) == 1: res = res * match
        else: assert len(res) == match, "Match length mismatch"
    return res

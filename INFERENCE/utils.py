from typing import Union, Tuple, List

size_2_t = Union[int, Tuple[int, int], List[int]]


def get_pair(x: size_2_t):
    if isinstance(x, int):
        return x, x
    assert len(x) == 2
    return tuple(x)

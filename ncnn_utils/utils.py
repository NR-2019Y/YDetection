def get_tuple_elems(x, num_elems: int):
    if isinstance(x, (list, tuple)):
        assert len(x) == num_elems
        return tuple(x)
    return (x,) * num_elems

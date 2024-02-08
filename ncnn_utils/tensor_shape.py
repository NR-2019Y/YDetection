from typing import Optional, Union, Tuple
import numpy as np


# reference: tensorflow-1.14.0 tensorflow/python/framework/tensor_shape.py

class Dimension:
    def __init__(self, value: Optional[int]):
        assert value is None or isinstance(value, int)
        self._value = value

    @property
    def value(self) -> Optional[int]:
        return self._value

    def __str__(self):
        return f"Dimension({self._value})"

    __repr__ = __str__

    def __eq__(self, other) -> Optional[bool]:
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return None
        return self._value == other._value

    def __ne__(self, other) -> Optional[bool]:
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return None
        return self._value != other._value

    def __lt__(self, other) -> Optional[bool]:
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return None
        return self._value < other._value

    def __le__(self, other) -> Optional[bool]:
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return None
        return self._value <= other._value

    def __gt__(self, other) -> Optional[bool]:
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return None
        return self._value > other._value

    def __ge__(self, other) -> Optional[bool]:
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return None
        return self._value >= other._value

    def __int__(self) -> int:
        return self._value

    def __index__(self) -> int:
        return self._value

    def __add__(self, other) -> 'Dimension':
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return Dimension(None)
        return Dimension(self._value + other._value)

    def __radd__(self, other) -> 'Dimension':
        return self + other

    def __sub__(self, other) -> 'Dimension':
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return Dimension(None)
        return Dimension(self._value - other._value)

    def __rsub__(self, other) -> 'Dimension':
        other = as_dimension(other)
        return other - self

    def __mul__(self, other) -> 'Dimension':
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return Dimension(None)
        return Dimension(self._value * other._value)

    def __rmul__(self, other) -> 'Dimension':
        return self * other

    def __floordiv__(self, other) -> 'Dimension':
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return Dimension(None)
        return Dimension(self._value // other._value)

    def __rfloordiv__(self, other) -> 'Dimension':
        other = as_dimension(other)
        return other // self

    def __mod__(self, other) -> 'Dimension':
        other = as_dimension(other)
        if self._value is None or other._value is None:
            return Dimension(None)
        return Dimension(self._value % other._value)

    def __rmod__(self, other) -> 'Dimension':
        other = as_dimension(other)
        return other % self

    def merge_with(self, other: 'Dimension') -> 'Dimension':
        assert isinstance(other, Dimension)
        if self.value is None:
            return Dimension(other.value)
        if other.value is None:
            return Dimension(self.value)
        assert self.value == other.value, "size do not match"
        return Dimension(self.value)

    def broadcast_with(self, other: 'Dimension') -> 'Dimension':
        assert isinstance(other, Dimension)
        if self.value is None:
            return Dimension(other.value)
        if other.value is None:
            return Dimension(self.value)
        if self.value == 1:
            return Dimension(other.value)
        if other.value == 1:
            return Dimension(self.value)
        assert self.value == other.value, "size do not match"
        return Dimension(self.value)


def as_dimension(d: Union[Dimension, int, None]):
    if isinstance(d, Dimension):
        return d
    return Dimension(d)


shape_type = Optional[Tuple[Dimension, ...]]


def shape_of_array(x: np.ndarray) -> Tuple[Dimension, ...]:
    return tuple(Dimension(d) for d in x.shape)

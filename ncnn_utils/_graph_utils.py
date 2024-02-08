from typing import Optional, List, Dict, Any, Type, Union
import contextlib
import functools
import operator
import io
import numpy as np

from .tensor_shape import Dimension, shape_type


class Graph:
    def __init__(self):
        self.nodes: List['Operation'] = []
        self.placeholders: List['Operation'] = []

    def add_node(self, node: 'Operation'):
        assert isinstance(node, Operation)
        self.nodes.append(node)

    def add_placeholder(self, op: 'Operation'):
        self.placeholders.append(op)


class _GraphMgr:
    __last_graph: Optional[Graph] = None

    @classmethod
    def set_last_graph(cls, g: Optional[Graph]):
        assert g is None or isinstance(g, Graph)
        cls.__last_graph = g

    @classmethod
    def get_last_graph(cls) -> Graph:
        return cls.__last_graph


get_default_graph = _GraphMgr.get_last_graph


@contextlib.contextmanager
def graph_scope(g: Graph):
    prev_graph: Optional[Graph] = get_default_graph()
    try:
        _GraphMgr.set_last_graph(g)
        yield g
    finally:
        _GraphMgr.set_last_graph(prev_graph)


param_dict_value_type = Union[int, float, List[int], List[float]]


class Operation:
    id: int
    op_type: str
    next_uid: int
    data_list: List[np.ndarray]

    @classmethod
    def get_uid(cls) -> int:
        uid = cls.next_uid
        cls.next_uid += 1
        return uid

    def __init__(self, inputs: List['Tensor'], param_dict: Dict[int, param_dict_value_type], num_outputs: int = 1,
                 output_shapes: Optional[List[shape_type]] = None):
        self.uid: int = self.get_uid()
        self.name: str = f"{self.op_type}:{self.uid}"
        self.inputs: List['Tensor'] = inputs
        if output_shapes is not None:
            assert num_outputs == len(output_shapes)
            for output_shape_i in output_shapes:
                if output_shape_i is not None:
                    assert all(isinstance(d, Dimension) for d in output_shape_i)
            outputs = [Tensor(self, i, shape=shape) for i, shape in enumerate(output_shapes)]
        else:
            outputs = [Tensor(self, i, shape=None) for i in range(num_outputs)]
        self.outputs: List['Tensor'] = outputs
        self.param_dict: Dict[int, param_dict_value_type] = param_dict
        g = get_default_graph()
        g.add_node(self)

    def write_weights(self, fp: io.BufferedIOBase):
        pass


class Tensor:
    def __init__(self, op: 'Operation', value_index: int = 0, shape: shape_type = None):
        self.op = op
        self.value_index = value_index
        self.shape = shape
        self.name = f"{self.op.name}:{self.value_index}"

    @property
    def size(self) -> Optional[int]:
        if self.shape is None:
            return None
        if any(d.value is None for d in self.shape):
            return None
        return functools.reduce(operator.mul, self.shape).value

    @property
    def ndim(self) -> Optional[int]:
        if self.shape is None:
            return None
        return len(self.shape)

    def dim_at(self, item: int) -> Dimension:
        if self.shape is None:
            return Dimension(None)
        return self.shape[item]

    def size_at(self, item: int) -> Optional[int]:
        return self.dim_at(item).value

    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        return _binary_func(0, self, other)

    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        return _binary_func(1, self, other)

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        return _binary_func(2, self, other)

    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        return _binary_func(3, self, other)

    def __pow__(self, other: Union['Tensor', float]) -> 'Tensor':
        return _binary_func(6, self, other)

    def __rsub__(self, other: Union['Tensor', float]) -> 'Tensor':
        return _binary_func(7, self, other)

    def __rtruediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        return _binary_func(8, self, other)

    def __rpow__(self, other: Union['Tensor', float]) -> 'Tensor':
        return _binary_func(9, self, other)

    def astype(self, dtype: str, from_type: str = "auto") -> 'Tensor':
        return _cast(self, dtype, from_type)


class RegisterOp:
    oplist: List[Type[Operation]] = []
    type2op: Dict[str, Type[Operation]] = dict()

    def __init__(self, op_type: Optional[str] = None):
        self.op_type = op_type

    def __call__(self, op_cls: Type[Operation]) -> Type[Operation]:
        op_type = self.op_type if self.op_type is not None else op_cls.__name__
        op_cls.id = len(self.oplist)
        op_cls.op_type = op_type
        op_cls.next_uid = 0
        self.oplist.append(op_cls)
        self.type2op[op_type] = op_cls
        return op_cls


def _compute_broadcast_shape(shape1: shape_type, shape2: shape_type) -> shape_type:
    if shape1 is None or shape2 is None:
        return None
    assert len(shape1) == len(shape2)
    return tuple(d1.broadcast_with(d2) for d1, d2 in zip(shape1, shape2))


def _binary_func(binop_type: int, x: Tensor, y: Union[Tensor, float]) -> Tensor:
    if isinstance(y, Tensor):
        param_dict = {0: binop_type, 1: 0}
        inputs = [x, y]
        output_shapes = [_compute_broadcast_shape(x.shape, y.shape)]
    else:
        assert isinstance(y, float)
        param_dict = {0: binop_type, 1: 1, 2: y}
        inputs = [x]
        output_shapes = [x.shape]
    op = BinaryOp(inputs, param_dict=param_dict, num_outputs=1, output_shapes=output_shapes)
    return op.outputs[0]


@RegisterOp("BinaryOp")
class BinaryOp(Operation):
    pass


@RegisterOp("Cast")
class Cast(Operation):
    pass


def _cast(x: Tensor, dtype: str, from_type: str = "auto"):
    type_dict = {
        "auto": 0,
        "float32": 1,
        "float16": 2,
        "int8": 3,
        "bfloat16": 4
    }
    assert dtype != "auto"
    param_dict = {0: type_dict[from_type], 1: type_dict[dtype]}
    op = Cast([x], param_dict=param_dict, num_outputs=1, output_shapes=[x.shape])
    return op.outputs[0]

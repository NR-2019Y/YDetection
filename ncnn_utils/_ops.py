from .tensor_shape import Dimension, shape_type, as_dimension
from ._graph_utils import Operation, Tensor, Graph, get_default_graph, RegisterOp, _binary_func
from typing import List, Dict, Optional, Union, Tuple
from enum import Enum
import io
import numpy as np
from .utils import get_tuple_elems


def _value_compatible(val1, val2) -> bool:
    if val1 is None or val2 is None:
        return True
    return val1 == val2


def _update_value(ori_value, new_value):
    if new_value is None:
        return ori_value
    if ori_value is not None:
        assert ori_value == new_value
    return new_value


@RegisterOp("AbsVal")
class AbsVal(Operation):
    pass


def abs_val(x: Tensor) -> Tensor:
    op = AbsVal([x], param_dict={}, num_outputs=1, output_shapes=[x.shape])
    return op.outputs[0]


@RegisterOp("ArgMax")
class ArgMax(Operation):
    pass


def argmax(x: Tensor, topk: int = 1, return_max: bool = False) -> Tensor:
    if return_max:
        output_shape = (Dimension(2), Dimension(topk))
    else:
        output_shape = (Dimension(1), Dimension(topk))
    op = ArgMax([x], param_dict={0: int(return_max), 1: topk}, num_outputs=1, output_shapes=[output_shape])
    return op.outputs[0]


def maximum(x: Tensor, y: Union[Tensor, float]) -> Tensor:
    return _binary_func(4, x, y)


def minimum(x: Tensor, y: Union[Tensor, float]) -> Tensor:
    return _binary_func(5, x, y)


def atan2(x: Union[Tensor, float], y: Union[Tensor, float]) -> Tensor:
    if isinstance(x, Tensor):
        return _binary_func(10, x, y)
    assert isinstance(x, float) and isinstance(y, Tensor)
    return _binary_func(11, y, x)


@RegisterOp("BNLL")
class BNLL(Operation):
    pass


def bnll(x: Tensor) -> Tensor:
    op = BNLL([x], param_dict={}, num_outputs=1, output_shapes=[x.shape])
    return op.outputs[0]


@RegisterOp("CELU")
class CELU(Operation):
    pass


def celu(x: Tensor, alpha: float = 1.0) -> Tensor:
    op = CELU([x], param_dict={0: float(alpha)}, num_outputs=1, output_shapes=[x.shape])
    return op.outputs[0]


@RegisterOp("Clip")
class Clip(Operation):
    pass


def clip(x: Tensor, _min: float = -3e38, _max: float = 3e38) -> Tensor:
    _min = max(_min, -3e38)
    _max = min(_max, 3e38)
    op = Clip([x], param_dict={0: _min, 1: _max}, num_outputs=1, output_shapes=[x.shape])
    return op.outputs[0]


@RegisterOp("Concat")
class Concat(Operation):
    pass


def _compute_concat_shape(x: List[Tensor], axis: int) -> shape_type:
    xiter = iter(x)
    x0 = next(xiter)
    ndim = x0.ndim
    for xi in xiter:
        if xi.ndim is not None:
            if ndim is None:
                ndim = xi.ndim
            else:
                assert ndim == xi.ndim, "ndim do not match"
    if ndim is None:
        return None
    xiter = iter(x)
    x0 = next(xiter)
    shape = x0.shape
    if shape is None:
        shape_list = [Dimension(None) for _ in range(ndim)]
    else:
        shape_list = list(shape)
    for xi in xiter:
        current_shape = xi.shape
        if current_shape is None:
            shape_list[axis] = Dimension(None)
        else:
            for i, (dprev, dcurr) in enumerate(zip(shape_list, current_shape)):
                if i == axis:
                    shape_list[i] += dcurr
                else:
                    shape_list[i] = dprev.merge_with(dcurr)
    return tuple(shape_list)


def concat(x: List[Tensor], axis: int = 0) -> Tensor:
    op = Concat(x, param_dict={0: axis}, num_outputs=1,
                output_shapes=[_compute_concat_shape(x, axis)])
    return op.outputs[0]


@RegisterOp("Convolution")
class Convolution(Operation):
    def write_weights(self, fp: io.BufferedIOBase):
        if not hasattr(self, "data_list"):
            return
        len_data = len(self.data_list)
        assert 1 <= len_data <= 2
        fp.write(bytes(4))
        for data in self.data_list:
            assert data.dtype == np.dtype("<f4")
            fp.write(data.tobytes())


def convolution(x: Tensor, weight: Union[Tensor, np.ndarray], bias: Union[Tensor, np.ndarray, None] = None,
                ksize: Union[None, int, Tuple[int, int]] = None,
                stride: Union[int, Tuple[int, int]] = 1,
                padding: Union[int, Tuple[int, int, int, int]] = 0,
                dilation: Union[int, Tuple[int, int]] = 1,
                pad_value: float = 0.,
                out_channels: Optional[int] = None,
                weight_data_size: Optional[int] = None) -> Tensor:
    """
    :param stride: [stride_h, stride_w]
    :param padding: [pad_left, pad_right, pad_top, pad_bottom]
    :param dilation: [dilation_h, dilation_w]
    """
    kernel_h, kernel_w = get_tuple_elems(ksize, 2)
    stride_h, stride_w = get_tuple_elems(stride, 2)
    pad_left, pad_right, pad_top, pad_bottom = get_tuple_elems(padding, 4)
    dilation_h, dilation_w = get_tuple_elems(dilation, 2)
    assert _value_compatible(x.ndim, 3), "input dim do not match!"
    assert weight.ndim == 4
    if bias is not None:
        assert bias.ndim == 1
    if isinstance(weight, np.ndarray):
        dynamic_weight = 0
        assert (bias is None) or isinstance(bias, np.ndarray)
        assert _value_compatible(x.size_at(0), weight.shape[1])
        if bias is not None:
            assert weight.shape[0] == bias.shape[0]
        weight_data_size = _update_value(weight_data_size, weight.size)
        out_channels = _update_value(out_channels, weight.shape[0])
        kernel_h = _update_value(kernel_h, weight.shape[2])
        kernel_w = _update_value(kernel_w, weight.shape[3])
    else:
        dynamic_weight = 1
        assert isinstance(weight, Tensor)
        assert bias is None or isinstance(bias, Tensor)
        assert _value_compatible(x.size_at(0), weight.size_at(1))
        if bias is not None:
            assert weight.size_at(0) == bias.size_at(0)
        weight_data_size = _update_value(weight_data_size, weight.size)
        if weight_data_size is None:
            raise ValueError("can not inference weight size!")
        out_channels = _update_value(out_channels, weight.size_at(0))
        if bias is not None:
            out_channels = _update_value(out_channels, bias.size_at(0))
        if out_channels is None:
            raise ValueError("can not inference output channels!")
        kernel_h = _update_value(kernel_h, weight.size_at(2))
        kernel_w = _update_value(kernel_w, weight.size_at(3))
        if kernel_h is None or kernel_w is None:
            raise ValueError("can not inference kernel size")
    input_shape = tuple(Dimension(None) for _ in range(3)) if x.shape is None else x.shape
    _, input_h, input_w = input_shape
    dkh = kernel_h + (kernel_h - 1) * (dilation_h - 1)
    dkw = kernel_w + (kernel_w - 1) * (dilation_w - 1)
    output_h = (input_h + pad_top + pad_bottom - dkh) // stride_h + 1
    output_w = (input_w + pad_left + pad_right - dkw) // stride_w + 1
    output_shape = (as_dimension(out_channels), as_dimension(output_h), as_dimension(output_w))
    param_dict = {
        0: out_channels,
        1: kernel_w,
        2: dilation_w,
        3: stride_w,
        4: pad_left,
        5: int(bias is not None),
        6: weight_data_size,
        11: kernel_h,
        12: dilation_h,
        13: stride_h,
        14: pad_top,
        15: pad_right,
        16: pad_bottom,
        18: float(pad_value),
        19: dynamic_weight
    }
    if not dynamic_weight:
        inputs = [x]
    else:
        if bias is not None:
            inputs = [x, weight]
        else:
            inputs = [x, weight, bias]
    op = Convolution(inputs, param_dict=param_dict, num_outputs=1, output_shapes=[output_shape])
    if not dynamic_weight:
        if bias is not None:
            op.data_list = [weight, bias]
        else:
            op.data_list = [weight]
    return op.outputs[0]


@RegisterOp("BatchNorm")
class BatchNorm(Operation):
    def write_weights(self, fp: io.BufferedIOBase):
        # src/layer/batchnorm.cpp
        # mb.load(xxx, 1)
        assert len(self.data_list) == 4
        for data in self.data_list:
            assert data.dtype == np.dtype('<f4')
            fp.write(data.tobytes())


def batch_norm(x: Tensor, bn_weight: np.ndarray, bn_bias: np.ndarray,
               running_mean: np.ndarray, running_var: np.ndarray, eps: float = 0.0):
    assert bn_weight.ndim == 1
    assert bn_bias.ndim == 1
    assert running_mean.ndim == 1
    assert running_var.ndim == 1
    assert bn_weight.size == bn_bias.size == running_mean.size == running_var.size
    out_channels = bn_weight.size
    assert _value_compatible(x.size_at(0), out_channels)
    if x.shape is None:
        output_shape = None
    elif x.size_at(0) is None:
        output_shape = (Dimension(out_channels),) + x.shape[1:]
    else:
        output_shape = x.shape
    op = BatchNorm([x], param_dict={0: out_channels, 1: float(eps)}, num_outputs=1, output_shapes=[output_shape])
    op.data_list = [bn_weight, running_mean, running_var, bn_bias]
    return op.outputs[0]


@RegisterOp("InnerProduct")
class InnerProduct(Operation):
    def write_weights(self, fp: io.BufferedIOBase):
        len_data = len(self.data_list)
        assert 1 <= len_data <= 2
        fp.write(bytes(4))
        for data in self.data_list:
            assert data.dtype == np.dtype("<f4")
            fp.write(data.tobytes())


def linear(x: Tensor, weight: np.ndarray, bias: Optional[np.ndarray]) -> Tensor:
    assert _value_compatible(x.ndim, 1) or _value_compatible(x.ndim, 2)
    assert weight.ndim == 2
    out_features, in_features = weight.shape
    if bias is not None:
        assert bias.shape == (out_features,)
    output_shape = None
    if x.shape is not None:
        assert _value_compatible(x.size_at(-1), in_features)
        if x.ndim == 1:
            output_shape = (Dimension(output_shape),)
        else:
            output_shape = (x.dim_at(0), Dimension(output_shape))
    op = InnerProduct([x], param_dict={
        0: out_features,
        1: 0 if bias is None else 1,
        2: in_features * out_features
    }, num_outputs=1, output_shapes=[output_shape])
    op.data_list = [weight] if bias is None else [weight, bias]
    return op.outputs[0]


@RegisterOp("Flatten")
class Flatten(Operation):
    pass


def flatten(x: Tensor) -> Tensor:
    op = Flatten([x], param_dict={}, num_outputs=1, output_shapes=[(Dimension(x.size),)])
    return op.outputs[0]


@RegisterOp("ReLU")
class ReLU(Operation):
    pass


def relu(x: Tensor, slope: float = 0.) -> Tensor:
    op = ReLU([x], param_dict={0: float(slope)}, num_outputs=1, output_shapes=[x.shape])
    return op.outputs[0]


@RegisterOp("Reorg")
class SpaceToDepth(Operation):
    pass


def space_to_depth(x: Tensor, stride: int, mode: int = 0):
    assert _value_compatible(x.ndim, 3)
    input_shape = tuple(Dimension(None) for _ in range(3)) if x.shape is None else x.shape
    in_channels, in_height, in_width = input_shape
    out_channels = in_channels * (stride * stride)
    assert _value_compatible((in_height % stride).value, 0)
    assert _value_compatible((in_width % stride).value, 0)
    out_height = in_height / stride
    out_width = in_width / stride
    output_shape = (out_channels, out_height, out_width)
    op = SpaceToDepth([x], param_dict={0: stride, 1: mode}, num_outputs=1, output_shapes=[output_shape])
    return op.outputs[0]


def _make_param_dict_from_shape(shape: Tuple[int, ...]) -> Dict[int, int]:
    assert 1 <= len(shape) <= 4
    if len(shape) == 1:
        # shape: [w]
        return {0: shape[0]}
    if len(shape) == 2:
        # shape: [h, w]
        return {0: shape[1], 1: shape[0]}
    if len(shape) == 3:
        # shape: [c, h, w]
        return {0: shape[2], 1: shape[1], 2: shape[0]}
    # shape: [c, d, h, w]
    return {0: shape[3], 1: shape[2], 2: shape[0], 11: shape[1]}


@RegisterOp("Input")
class Input(Operation):
    def __init__(self, shape: Optional[Tuple[Optional[int], ...]] = None):
        param_dict = {}
        input_shape: shape_type = None
        if shape is not None:
            assert 1 <= len(shape) <= 4
            input_shape = tuple(Dimension(d) for d in shape)
        if (shape is not None) and all(d is not None for d in shape):
            param_dict = _make_param_dict_from_shape(shape)
        super(Input, self).__init__([], param_dict=param_dict, num_outputs=1, output_shapes=[input_shape])
        g: Graph = get_default_graph()
        g.add_placeholder(self)


def placeholder(shape: Optional[Tuple[Optional[int], ...]] = None) -> Tensor:
    op = Input(shape)
    return op.outputs[0]


@RegisterOp("Bias")
class Bias(Operation):
    def write_weights(self, fp: io.BufferedIOBase):
        data = self.data_list[0]
        assert data.dtype == np.dtype('<f4')
        fp.write(data.tobytes())


def add_bias(x: Tensor, bias: np.ndarray) -> Tensor:
    assert _value_compatible(x.ndim, 3) or _value_compatible(x.ndim, 4)
    assert bias.ndim == 1
    assert _value_compatible(x.size_at(0), bias.size)
    if x.size_at(0) is not None:
        output_shape = x.shape
    elif x.shape is None:
        output_shape = None
    else:
        output_shape = (Dimension(bias.size),) + x.shape[1:]
    op = Bias([x], param_dict={0: bias.size}, num_outputs=1, output_shapes=[output_shape])
    op.data_list = [bias]
    return op.outputs[0]


@RegisterOp("MemoryData")
class MemoryData(Operation):
    def write_weights(self, fp: io.BufferedIOBase):
        data = self.data_list[0]
        assert data.dtype == np.dtype('<f4')
        fp.write(data.tobytes())


def make_tensor(x: np.ndarray) -> Tensor:
    assert isinstance(x, np.ndarray)
    assert 1 <= x.ndim <= 4
    shape = tuple(Dimension(d) for d in x.shape)
    param_dict = _make_param_dict_from_shape(x.shape)
    op = MemoryData([], param_dict=param_dict, num_outputs=1, output_shapes=[shape])
    op.data_list = [x]
    return op.outputs[0]


@RegisterOp("Split")
class Split(Operation):
    pass


def split(x: Tensor, count: int) -> List[Tensor]:
    op = Split([x], param_dict={}, num_outputs=count, output_shapes=[x.shape for _ in range(count)])
    return op.outputs


@RegisterOp("Pooling")
class Pooling(Operation):
    pass


class PoolingMode(Enum):
    MAX = 0
    AVG = 1


def pool2d(x: Tensor,
           mode: PoolingMode,
           ksize: Union[int, Tuple[int, int]],
           stride: Union[int, Tuple[int, int], None],
           padding: Union[int, Tuple[int, int, int, int]] = 0,
           ceil_mode: bool = False) -> Tensor:
    # ceil_mode: 不完全等同于pytorch，表示full padding模式
    kernel_h, kernel_w = get_tuple_elems(ksize, 2)
    stride_h, stride_w = get_tuple_elems(stride, 2)
    pad_left, pad_right, pad_top, pad_bottom = get_tuple_elems(padding, 4)
    assert _value_compatible(x.ndim, 3)
    input_shape = tuple(Dimension(None) for _ in range(3)) if x.shape is None else x.shape
    in_channels, in_height, in_width = input_shape
    if ceil_mode:
        out_height = (in_height + pad_top + pad_bottom - kernel_h + stride_h - 1) // stride_h + 1
        out_width = (in_width + pad_left + pad_right - kernel_w + stride_w - 1) // stride_w + 1
    else:
        out_height = (in_height + pad_top + pad_bottom - kernel_h) // stride_h + 1
        out_width = (in_width + pad_left + pad_right - kernel_w) // stride_w + 1
    output_shape = (as_dimension(in_channels), as_dimension(out_height), as_dimension(out_width))
    param_dict = {
        0: mode.value,
        1: kernel_w,
        2: stride_w,
        3: pad_left,
        5: 0 if ceil_mode else 1,
        11: kernel_h,
        12: stride_h,
        13: pad_top,
        14: pad_right,
        15: pad_bottom
    }
    op = Pooling([x], param_dict=param_dict, num_outputs=1, output_shapes=[output_shape])
    return op.outputs[0]


def maxpool2d(x: Tensor,
              ksize: Union[int, Tuple[int, int]],
              stride: Union[int, Tuple[int, int], None],
              padding: Union[int, Tuple[int, int, int, int]] = 0,
              ceil_mode: bool = False) -> Tensor:
    return pool2d(x, PoolingMode.MAX, ksize, stride, padding, ceil_mode)


def avgpool2d(x: Tensor,
              ksize: Union[int, Tuple[int, int]],
              stride: Union[int, Tuple[int, int], None],
              padding: Union[int, Tuple[int, int, int, int]] = 0,
              ceil_mode: bool = False) -> Tensor:
    return pool2d(x, PoolingMode.AVG, ksize, stride, padding, ceil_mode)


def global_pool2d(x: Tensor, mode: PoolingMode) -> Tensor:
    assert _value_compatible(x.ndim, 3)
    output_shape = (x.dim_at(0),)
    param_dict = {0: mode.value, 4: 1}
    op = Pooling([x], param_dict=param_dict, num_outputs=1, output_shapes=[output_shape])
    return op.outputs[0]


def global_maxpool2d(x: Tensor) -> Tensor:
    return global_pool2d(x, PoolingMode.MAX)


def global_avgpool2d(x: Tensor) -> Tensor:
    return global_pool2d(x, PoolingMode.AVG)


def adaptive_pool2d(x: Tensor, mode: PoolingMode, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    assert _value_compatible(x.ndim, 3)
    output_h, output_w = get_tuple_elems(output_size, 2)
    output_shape = (x.dim_at(0), Dimension(output_h), Dimension(output_w))
    param_dict = {0: mode.value, 7: 1, 8: output_w, 18: output_h}
    op = Pooling([x], param_dict=param_dict, num_outputs=1, output_shapes=[output_shape])
    return op.outputs[0]


def adaptive_maxpool2d(x: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    return adaptive_pool2d(x, PoolingMode.MAX, output_size)


def adaptive_avgpool2d(x: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    return adaptive_pool2d(x, PoolingMode.AVG, output_size)

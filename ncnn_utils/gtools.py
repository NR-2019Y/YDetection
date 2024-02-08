import itertools
from . import _ops as F
from typing import Dict, List, Sequence, Set, NamedTuple, Tuple, Optional
from ._graph_utils import Graph, Operation, Tensor, graph_scope
from collections import defaultdict


def topological_sort(graph: Dict[Operation, List[Operation]],
                     start_nodes: Sequence[Operation]) -> List[Operation]:
    ord_nodes: List[Operation] = []
    visited: Set[Operation] = set()
    q: List[Tuple[Operation, int]] = []
    curr_path: List[Operation] = []
    for op in start_nodes:
        if op in visited:
            continue
        q.append((op, 0))
        while len(q):
            last_op, layer = q.pop()
            while len(curr_path) > layer:
                ord_nodes.append(curr_path.pop())
            if last_op in visited:
                continue
            visited.add(last_op)
            curr_path.append(last_op)
            for c_op in graph[last_op]:
                if c_op not in visited:
                    q.append((c_op, layer + 1))
        while len(curr_path):
            ord_nodes.append(curr_path.pop())
    return ord_nodes


class NCNNModel(NamedTuple):
    ord_nodes: List[Operation]
    inputs: List[Operation]
    outputs: List[Operation]

    @staticmethod
    def from_graph(g: Graph,
                   outputs: Optional[Sequence[Tensor]] = None) -> 'NCNNModel':
        # add Split
        tensor_2_nodes: Dict[Tensor, List[Tuple[Operation, int]]] = defaultdict(list)
        for op in g.nodes:
            for i, t in enumerate(op.inputs):
                tensor_2_nodes[t].append((op, i))

        with graph_scope(g):
            for t, ops in tensor_2_nodes.items():
                if len(ops) < 2:
                    continue
                t_splited: List[Tensor] = F.split(t, len(ops))
                for ti, (op, i) in zip(t_splited, ops):
                    op.inputs[i] = ti

        edges: Set[Tuple[Operation, Operation]] = set()
        # graph_ori: Dict[Operation, List[Operation]] = dict()
        node_out_degree: Dict[Operation, int] = defaultdict(int)
        graph_transpose: Dict[Operation, List[Operation]] = defaultdict(list)

        for op in g.nodes:
            # graph_ori[op] = []
            # graph_transpose[op] = []
            for tin in op.inputs:
                edges.add((tin.op, op))

        for op_from, op_to in edges:
            # graph_ori[op_from].append(op_to)
            node_out_degree[op_from] += 1
            graph_transpose[op_to].append(op_from)

        if (outputs is None) or (len(outputs) == 0):
            output_ops: List[Operation] = []
            for op, degree in node_out_degree.items():
                if degree == 0:
                    output_ops.append(op)
            assert len(output_ops) != 0
        else:
            output_ops: List[Operation] = list(set(t.op for t in outputs))

        ord_nodes: List[Operation] = topological_sort(graph_transpose, output_ops)
        return NCNNModel(ord_nodes=ord_nodes, inputs=g.placeholders, outputs=output_ops)

    def to_file(self, prefix: str):
        blobs: Set[Tensor] = set()
        for op in self.ord_nodes:
            blobs.update(op.inputs)
            blobs.update(op.outputs)
        num_blobs = len(blobs)
        num_layers = len(self.ord_nodes)
        param_path = f"{prefix}.param"
        with open(param_path, "w") as fp:
            print(f"7767517\n{num_layers} {num_blobs}", file=fp)
            for op in self.ord_nodes:
                print(' '.join(itertools.chain(
                    (op.op_type, op.name, str(len(op.inputs)), str(len(op.outputs))),
                    (t.name for t in op.inputs),
                    (t.name for t in op.outputs)
                )), end="" if op.param_dict else "\n", file=fp)
                if op.param_dict:
                    for k, v in op.param_dict.items():
                        fp.write(' ')
                        if isinstance(v, int):
                            fp.write(f"{k}={v}")
                        elif isinstance(v, float):
                            fp.write(f"{k}={v:e}")
                        else:
                            assert isinstance(v, list)
                            fp.write(str(len(v)))
                            assert isinstance(v[0], int) or isinstance(v[0], float)
                            assert all(isinstance(vi, type(v[0])) for vi in v)
                            if isinstance(v[0], int):
                                for vi in v:
                                    fp.write(f",{v}")
                            else:
                                for vi in v:
                                    fp.write(f",{v:e}")
                    fp.write("\n")
        model_path = f"{prefix}.bin"
        with open(model_path, "wb") as fp:
            for op in self.ord_nodes:
                if hasattr(op, "data_list"):
                    op.write_weights(fp)  # type: ignore

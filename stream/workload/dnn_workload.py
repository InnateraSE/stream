import logging
from typing import Any
from copy import deepcopy

from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


class DNNWorkloadStream(ComputationNodeWorkload):
    """
    Collect all the algorithmic workload information here.
    Similar to `DNNWorkload` from ZigZag, but returns a DiGraph of ComputationNodes instead of LayerNodes.

    :return (self): Directed Graph with nodes the layers and edges the connections between layers.
    """
    def __init__(self, nodes: list[ComputationNode], **attr: Any):
        super().__init__(**attr)  # type: ignore

        cn_id_to_obj: dict[int, ComputationNode] = {}
        self.layer_node_list = nodes

        for cn in nodes:
            cn_id_to_obj[cn.id] = cn

            self.add_node(cn)
            # Find all of its operand sources and add edges accordingly
            edges: list[tuple[ComputationNode, ComputationNode]] = []
            for _, parent_id in cn.input_operand_source.items():
                # for parent_id in parent_list:
                assert parent_id in cn_id_to_obj, f"Illegal reference to non-existent layer with id {parent_id}"
                parent_layer = cn_id_to_obj[parent_id]
                edges.append((parent_layer, cn))

            self.add_edges_from(edges)

    def get_copy_no_dummy(self) -> "DNNWorkloadStream":
        """Return a copy. DNNWorkloads don't contain DummyNodes in the first place."""
        return deepcopy(self)

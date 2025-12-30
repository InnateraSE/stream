from zigzag.parser.workload_factory import LayerNodeFactory
from zigzag.workload.layer_node import LayerNode

from stream.workload.dnn_workload import DNNWorkloadStream
from stream.workload.mapping import InterCoreMappingAttributes
from stream.workload.computation.computation_node import ComputationNode
from stream.hardware.architecture.accelerator import Accelerator

class WorkloadFactoryStream:
    """Generates a `Workload` instance from the validated and normalized user-provided data.
    Almost identical to ZigZagWorkloadFactory, apart from the return type: DNNWorkloadStream instead of
    DNNWorkload
    """
    def __init__(self, workload_data: list[dict], all_mappings: dict[str, InterCoreMappingAttributes],
                 accelerator: Accelerator):
        self.workload_data = workload_data
        self.all_mappings = all_mappings
        self.accelerator = accelerator

    def create(self) -> DNNWorkloadStream:  # type: ignore
        node_list: list[ComputationNode] = []
        for layer_data in self.workload_data:
            # TODO: don't create layer note but only extract the attributes
            layer_node_factory = LayerNodeFactory(layer_data, mapping_data=[])
            node_attr = layer_node_factory.create_node_attr()
            node_id = layer_node_factory.layer_id
            node_name = layer_node_factory.node_name
            mapping = self.get_mapping(node_id, node_name, node_attr.layer_type)
            comp_node = ComputationNode(node_id=node_id, node_name=node_name,
                                       op_type=node_attr.layer_type, node_attr=node_attr,
                                       mapping_attr=mapping, input_names=[])
            node_list.append(comp_node)

        return DNNWorkloadStream(node_list)

    def get_mapping(self, node_id, node_name, node_type) -> InterCoreMappingAttributes:
        """Get the mapping that corresponds to this node's operator. Replace the spatial mapping with the corresponding
        core's dataflows.
        NOTE The core's dataflow always precedes the mapping's spatial mapping
        TODO Mapping based on node name instead of note operator is not yet supported
        """
        default_mapping = self.all_mappings["default"]
        if node_id in self.all_mappings:
            mapping = self.all_mappings[node_id]
        elif node_name in self.all_mappings:
            mapping = self.all_mappings[node_name]
        elif node_type in self.all_mappings:
            mapping = self.all_mappings[node_type]
        else:
            mapping = default_mapping

        # Override spatial mapping by the one defined in the core's dataflows
        try:
            core_dataflow = self.accelerator.get_spatial_mapping_from_core(mapping.core_allocation)
            mapping.spatial_mapping = core_dataflow
        except ValueError:
            pass

        # If no inter/intra mapping is given: use default one
        if not mapping.intra_core_tiling:
            mapping.intra_core_tiling = default_mapping.intra_core_tiling
        if not mapping.inter_core_tiling:
            mapping.inter_core_tiling = default_mapping.inter_core_tiling

        return mapping

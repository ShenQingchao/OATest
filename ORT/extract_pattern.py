import onnx
import networkx as nx
from collections import defaultdict


def build_dependency_graph(onnx_model):
    graph = nx.DiGraph()
    tensor_to_operators = defaultdict(list)
    node_map = {node.name: node for node in onnx_model.graph.node}  # Map node names to NodeProto
    node_map_ini = {initializer.name: initializer for initializer in onnx_model.graph.initializer}
    node_map.update(node_map_ini)
    # print(node_map.keys())
    # print(node_map_ini.keys())

    # Map operators to their input/output tensors
    for node in onnx_model.graph.node:
        for input_tensor in node.input:
            tensor_to_operators[input_tensor].append((node.name, 'input'))
        for output_tensor in node.output:
            tensor_to_operators[output_tensor].append((node.name, 'output'))

    # Add edges to the graph based on tensor flow
    for tensor, operators in tensor_to_operators.items():
        for op1 in operators:
            for op2 in operators:
                if op1[0] != op2[0] and op1[1] == 'output' and op2[1] == 'input':
                    graph.add_edge(op1[0], op2[0])  # Directed edge from producer to consumer
    return graph, node_map


def extract_dataflows(graph, node_map):
    """Extract dataflows by grouping operators based on shared tensors."""
    visited = set()
    dataflows = []

    for node in graph.nodes:
        if node not in visited:
            # Perform a DFS to identify a dataflow
            dataflow = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    dataflow.add(current)
                    stack.extend(neighbor for neighbor in graph.successors(current))
            # Convert node names to ONNX NodeProto
            dataflows.append([node_map[name] for name in dataflow])
    return dataflows


def extract_blocks(dataflows, graph, node_map):
    """Extract blocks from each dataflow."""
    blocks = []
    for dataflow in dataflows:
        # Create a subgraph for the dataflow
        subgraph = graph.subgraph([node.name for node in dataflow])  # Use node names for subgraph
        # Use a topological sort to find execution order
        sorted_nodes = list(nx.topological_sort(subgraph))
        # Initialize block grouping
        dataflow_blocks = []  # Blocks for the current dataflow
        current_block = []  # Current block being constructed
        seen_tensors = set()
        for node_name in sorted_nodes:
            # Check if the node can share tensors with the current block
            has_shared_tensor = any(
                tensor in seen_tensors for tensor in subgraph.predecessors(node_name)
            )
            if not has_shared_tensor and current_block:
                # Finalize the current block and start a new one
                dataflow_blocks.append([node_map[name] for name in current_block])
                current_block = []
            # Add the node to the current block
            current_block.append(node_name)
            # Update seen tensors with the outputs of the current node
            seen_tensors.update(subgraph.successors(node_name))
        # Add the last block if not empty
        if current_block:
            dataflow_blocks.append([node_map[name] for name in current_block])
        blocks.append(dataflow_blocks)
    return blocks


def extract_all_blocks(onnx_model):
    dependency_graph, node_map = build_dependency_graph(onnx_model)
    dataflows = extract_dataflows(dependency_graph, node_map)
    blocks = extract_blocks(dataflows, dependency_graph, node_map)

    blocks_list = []
    for dataflow_blocks in blocks:
        for block in dataflow_blocks:
            blocks_list.append(block)
    return blocks_list


def extract_dataflows_and_blocks(onnx_model):
    dependency_graph, node_map = build_dependency_graph(onnx_model)
    dataflows = extract_dataflows(dependency_graph, node_map)
    blocks = extract_blocks(dataflows, dependency_graph, node_map)
    return dataflows, blocks


if __name__ == "__main__":
    model_path = "tests/target_model.onnx"
    onnx_model = onnx.load(model_path)
    dataflows, blocks = extract_dataflows_and_blocks(onnx_model)

    print("Dataflows:")
    for i, dataflow in enumerate(dataflows):
        print(f"Dataflow {i + 1}: {[node.name for node in dataflow]}")

    print("\nBlocks:")
    for i, dataflow_blocks in enumerate(blocks):
        print(f"Dataflow {i + 1}:")
        for j, block in enumerate(dataflow_blocks):
            print(f"  Block {j + 1}: {[node.name for node in block]}")

import onnx


def get_all_vars(model):
    var_names = set()

    for input_tensor in model.graph.input:
        var_names.add(input_tensor.name)

    for node in model.graph.node:
        var_names.update(node.output)

    for output_tensor in model.graph.output:
        var_names.add(output_tensor.name)

    seed_node_names = [node.name for node in model.graph.node]
    var_names.update(seed_node_names)
    return var_names


def rename_onnx_variables(seed_model, donor_model):
    rename_map = {}

    def get_unique_name(name):
        if name not in seed_var_names:
            return name
        idx = 1
        new_name = f"{name}_{idx}"
        while new_name in seed_var_names:
            idx += 1
            new_name = f"{name}_{idx}"
        return new_name

    seed_var_names = get_all_vars(seed_model)
    # print(seed_var_names)

    # Rename the I/O and node name for each node
    for node in donor_model.graph.node:
        node.name = get_unique_name(node.name)  # remove duplicated node name
        node.input[:] = [rename_map.setdefault(i, get_unique_name(i)) for i in node.input]
        node.output[:] = [rename_map.setdefault(o, get_unique_name(o)) for o in node.output]
    
    # Rename the I/O of the graph
    for input_tensor in donor_model.graph.input:
        input_tensor.name = rename_map.setdefault(input_tensor.name, get_unique_name(input_tensor.name))
    for output_tensor in donor_model.graph.output:
        output_tensor.name = rename_map.setdefault(output_tensor.name, get_unique_name(output_tensor.name))
    # onnx.checker.check_model(donor_model)
    
    # Rename the value_info in graph
    for value_info in donor_model.graph.value_info:
        value_info.name = rename_map.setdefault(value_info.name, get_unique_name(value_info.name))

    # save the renamed model
    output_path = '_temp_renamed_model.onnx'
    # print(donor_model)
    onnx.save(donor_model, output_path)
    onnx.checker.check_model(donor_model)
    return donor_model

import onnx
from onnx import shape_inference
import copy
import random
import utils
from preprocess import rename_onnx_variables
import add_auxiliary_layer
from utils import merge_opset_imports
random.seed(2024)


def random_insert_position(seed_model):
    nodes = seed_model.graph.node
    return random.randint(0, len(nodes))


def modify_donor_model(donor_model, seed_out_shape, seed_out_type):
    for inp in donor_model.graph.input:
        del inp.type.tensor_type.shape.dim[:]

        for dim in seed_out_shape.dim:
            new_dim = inp.type.tensor_type.shape.dim.add()
            if dim.HasField("dim_value"):
                new_dim.dim_value = dim.dim_value
            elif dim.HasField("dim_param"):
                new_dim.dim_param = dim.dim_param

        inp.type.tensor_type.elem_type = seed_out_type.tensor_type.elem_type
    return donor_model


def collect_contexts_vars(model, insert_point):
    tensor_dict = {tensor.name: tensor for tensor in model.graph.value_info}
    all_vars_list = [[], []]  # [[preceding context], [following context]]
    for item in model.graph.input:
        if utils.var_with_shape(item):  # skip the var without shape
            all_vars_list[0].append(item)

    all_nodes = model.graph.node
    for i, node in enumerate(all_nodes):
        if i < insert_point:  # preceding context for "output"
            for _var in node.output:
                if _var in tensor_dict:
                    if utils.var_with_shape(_var):
                        all_vars_list[0].append(tensor_dict[_var])
        elif i == insert_point:  # todo: check it!
            continue
        else:  # following context for "inputs"
            for _var in node.input:
                if _var in tensor_dict:
                    if utils.var_with_shape(_var):
                        all_vars_list[1].append(tensor_dict[_var])

    for item in model.graph.output:
        if utils.var_with_shape(item):
            all_vars_list[1].append(item)

    return all_vars_list


def get_undefined_var(inputs, undefined_vars):
    undefined_vars = {var.name: var for var in undefined_vars}
    all_undefined_vars_dict = {}
    for _input in inputs:
        if _input in undefined_vars.keys():
            all_undefined_vars_dict[_input] = undefined_vars[_input]
    return all_undefined_vars_dict


def choose_var_from_context(context_vars_list, target_var_constraint):
    for seed_var in context_vars_list:
        this_var_constraint = utils.get_var_constraint(seed_var)
        if this_var_constraint[2] == target_var_constraint[2]:
            if this_var_constraint[1] == target_var_constraint[1]:
                return seed_var
    if len(context_vars_list):
        return random.choice(context_vars_list)
    return None


def insert_donor_into_seed(seed_model, donor_model, insert_position):
    context_vars_list = collect_contexts_vars(seed_model, insert_position)

    # create new connection for the preceding context
    insert_node_num = 0
    all_preceding_undefined_var_num = 0
    for node in donor_model.graph.node:
        undefined_var_dict = get_undefined_var(node.input, donor_model.graph.input)
        all_preceding_undefined_var_num += len(undefined_var_dict)
        if len(undefined_var_dict) != 0:
            node_new_input = []
            for old_input in node.input:
                if old_input in undefined_var_dict.keys():
                    this_donor_input_var = undefined_var_dict[old_input]
                    donor_input_constraint = utils.get_var_constraint(this_donor_input_var)
                    #
                    if len(context_vars_list[0]) == 0:  # none preceding context
                        # Add as the donor's input into seed input
                        for this_input in donor_model.graph.input:
                            if this_input.name == old_input:
                                seed_model.graph.input.append(this_input)
                        # seed_model.graph.node.insert(insert_position + insert_node_num, node)
                        continue

                    connection_point_var = choose_var_from_context(context_vars_list[0], donor_input_constraint)
                    if not connection_point_var:
                        return None
                    connect_var_name, connect_var_dtype, connect_var_shape = utils.get_var_constraint(connection_point_var)

                    if connect_var_dtype != donor_input_constraint[1]:  # retype
                        aux_layer, aux_out_name = add_auxiliary_layer.create_retype_layer(connect_var_name, donor_input_constraint[1])
                        seed_model.graph.node.insert(insert_position+insert_node_num, aux_layer)
                        insert_node_num += 1
                        connect_var_name = aux_out_name

                    if connect_var_shape != donor_input_constraint[2]:  # reshape
                        reshape_padding_nodes, auxiliary_initializers, reshape_output_name = add_auxiliary_layer.create_reshape_layer(
                            connect_var_name, connect_var_shape, donor_input_constraint[2], donor_input_constraint[1])
                        seed_model.graph.initializer.extend(auxiliary_initializers)

                        for cnt, this_auxiliary_node in enumerate(reshape_padding_nodes):
                            this_insert_position = insert_position+insert_node_num+cnt
                            seed_model.graph.node.insert(this_insert_position, this_auxiliary_node)
                        insert_node_num += len(reshape_padding_nodes)
                        node_new_input.append(reshape_output_name)
                    else:
                        node_new_input.append(connect_var_name)
                else:
                    node_new_input.append(old_input)
            new_node = copy.deepcopy(node)
            new_node.input[:] = node_new_input
            if len(context_vars_list[0]) == 0:
                seed_model.graph.node.insert(insert_position + insert_node_num, node)
            else:
                seed_model.graph.node.insert(insert_position+insert_node_num, new_node)
        else:
            seed_model.graph.node.insert(insert_position+insert_node_num, node)
        insert_node_num += 1

    # create new connection for the following context
    following_var_map = {}
    donor_outputs = donor_model.graph.output
    all_seed_output_name = [o.name for o in seed_model.graph.output]
    for donor_out in donor_outputs:
        donor_out_name = donor_out.name
        this_donor_out_var_constraint = utils.get_var_constraint(donor_out)

        # the following connection node is the 'output' directly
        if len(context_vars_list[1]) == 0:
            seed_model.graph.output.append(donor_out)
        following_connection_var = choose_var_from_context(context_vars_list[1], this_donor_out_var_constraint)
        if following_connection_var is None:
            return None
        if following_connection_var.name in all_seed_output_name:
            seed_model.graph.output.append(donor_out)
            if all_preceding_undefined_var_num == 0:  # not preceding & following connection!
                return donor_model
        else:
            # replace the donor_out with the following_connection_var
            connect_var_name, connect_var_dtype, connect_var_shape = utils.get_var_constraint(following_connection_var)
            if this_donor_out_var_constraint[1] != connect_var_dtype:  # retype
                aux_layer, aux_out_name = add_auxiliary_layer.create_retype_layer(donor_out_name, donor_input_constraint[1])
                seed_model.graph.node.insert(insert_position+insert_node_num, aux_layer)
                insert_node_num += 1
                connect_var_name = aux_out_name
            if this_donor_out_var_constraint[2] != connect_var_shape:  # reshape
                reshape_padding_nodes, auxiliary_initializers, reshape_output_name = add_auxiliary_layer.create_reshape_layer(
                    connect_var_name, connect_var_shape, this_donor_out_var_constraint[2], connect_var_dtype)
                seed_model.graph.initializer.extend(auxiliary_initializers)

                for this_id, this_auxiliary_node in enumerate(reshape_padding_nodes):
                    this_insert_position = insert_position + insert_node_num + this_id
                    # import pdb;pdb.set_trace()
                    seed_model.graph.node.insert(this_insert_position, this_auxiliary_node)
                donor_out_name = reshape_output_name
            if following_connection_var.name not in following_var_map.keys():
                following_var_map[following_connection_var.name] = []
            following_var_map[following_connection_var.name].append(donor_out_name)
        # print(f"[DEBUG] following_var_map:{following_var_map}")

    def get_the_connect_var_point(_node, all_target_var):
        this_var_list = []
        for _input in _node.input:
            if _input in all_target_var:
                this_var_list.append(_input)
        return this_var_list

    for node_id, node in enumerate(seed_model.graph.node):
        if node_id <= insert_position+insert_node_num:
            continue
        connected_vars_in_this_node = get_the_connect_var_point(node, following_var_map.keys())
        # print(node.input)
        if len(connected_vars_in_this_node):
            # import pdb;pdb.set_trace()  # todo: support more
            node_new_input = [_var if _var not in connected_vars_in_this_node else random.choice(following_var_map[_var])
                              for _var in node.input]
            seed_model.graph.node[node_id].input[:] = node_new_input
            # print(node_new_input)

    seed_initializer_constant_name_list = [v.name for v in seed_model.graph.initializer]
    donor_initializer_constant = donor_model.graph.initializer
    for tmp_var in donor_initializer_constant:
        if tmp_var.name not in seed_initializer_constant_name_list:
            seed_model.graph.initializer.append(tmp_var)
        else:  # todo: add the duplicate var name
            pass
    # seed_model.graph.initializer.extend(donor_initializer_constant)
    # for tmp in seed_model.graph.initializer:
    #     print(tmp.name)

    # merge the opset_import from two models.
    merged_opset_imports = merge_opset_imports(seed_model, donor_model)
    del seed_model.opset_import[:]
    for opset in merged_opset_imports:
        seed_model.opset_import.append(opset)
    return seed_model


def combine_models_random(seed_model, donor_model, position_flag='random'):
    donor_model = rename_onnx_variables(seed_model, donor_model)
    try:
        donor_model = onnx.version_converter.convert_version(donor_model, 21)
        seed_model = onnx.version_converter.convert_version(seed_model, 21)
    except Exception as e:
        print(f"[ONNXConversion Failed] {e}")
        return None
    # onnx.checker.check_model(seed_model， full_check=True)
    # # onnx.save(donor_model, "renamed_model.onnx")

    if position_flag == 'random':
        insert_position = random_insert_position(seed_model)
        combined_model = insert_donor_into_seed(seed_model, donor_model, insert_position)
        if combined_model is None:
            return None

    # print(combined_model.graph.node[0])
    # onnx.save(combined_model, "new.onnx")  # "just for debug"
    combined_model = shape_inference.infer_shapes(combined_model)
    try:
        onnx.checker.check_model(combined_model, full_check=True)
    except Exception as e:
        if "in initializer but not in graph input" in str(e):
            pass
        else:
            assert False, f"[Invalid Model]: {e}"
    # print(onnx.helper.printable_graph(combined_model.graph))
    # import os
    # os.system("python test_ort.py new.onnx")  # "just for debug"
    print("[INFO] Generate a valid ONNX model!")
    return combined_model


if __name__ == '__main__':
    seed_model = onnx.load("../res/onnx_ut/onnx_models_gf/MatMulAddFusion/MatMulAddFusion_10.onnx")
    donor_model = onnx.load("../res/onnx_ut/onnx_models_gf/RemoveDuplicateCastTransformer/RemoveDuplicateCastTransformer_77.onnx")
    for i in range(1):
        combine_models_random(seed_model, donor_model, position_flag='random')

from onnx import helper


def shape2list(onnx_shape):
    shape_list = []
    for dim in onnx_shape.dim:
        if dim.HasField("dim_value"):
            shape_list.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape_list.append(dim.dim_param)
    return shape_list


def dynamic2static(shape):
    new_shape = [1 if isinstance(dim, str) else dim for dim in shape]
    return new_shape


def get_var_constraint(onnx_var):
    tensor_type = onnx_var.type.tensor_type
    res_dict = [onnx_var.name, tensor_type.elem_type, dynamic2static(shape2list(tensor_type.shape))]
    return res_dict


def var_with_shape(onnx_var):
    if isinstance(onnx_var, str):
        return False
    tensor_type = onnx_var.type.tensor_type
    if not hasattr(tensor_type, "shape"):
        return False
    else:
        if dynamic2static(shape2list(tensor_type.shape)):
            return True
        else:
            return False


def merge_opset_imports(model1, model2):
    opset_map = {}

    def _isReduceinGraph(model):
        # if ReduceXX op in model, we should keep it in the origin model; due diff opset have differ attributes
        for node in model.graph.node:
            if "Reduce" in node.name:
                return True
        return False
    model1_included_reduce_op = _isReduceinGraph(model1)
    model2_included_reduce_op = _isReduceinGraph(model2)

    for opset in model1.opset_import:
        opset_map[opset.domain] = opset.version
    for opset in model2.opset_import:
        if opset.domain in opset_map:
            # if opset.domain == '' and model1_included_reduce_op:  # keep the origin opset version
            #     continue
            # if opset.domain == '' and model2_included_reduce_op:
            #     opset_map[opset.domain] = opset.version
            # else:
            #     opset_map[opset.domain] = max(opset_map[opset.domain], opset.version)
            opset_map[opset.domain] = max(opset_map[opset.domain], opset.version)
        else:
            opset_map[opset.domain] = opset.version

    merged_opset_imports = [
        helper.make_opsetid(domain=domain, version=version)
        for domain, version in opset_map.items()
    ]
    return merged_opset_imports


if __name__ == '__main__':
    import onnx
    m = onnx.load('tests/seed.onnx')
    res = get_var_constraint(m.graph.input[0])
    print(res)


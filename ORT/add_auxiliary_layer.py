import onnx
from onnx import helper, numpy_helper
import numpy as np
import random
import string
from utils import dynamic2static
import type_utils


def _get_random_id():
    # return np.random.randint(1, 1000)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=3))


def create_retype_layer(var_name, target_type):
    output_name = var_name + f"_{_get_random_id()}"
    cast_node = helper.make_node(
        "Cast",
        inputs=[var_name],
        outputs=[output_name],
        name=f"cast{_get_random_id()}",
        to=target_type
    )
    return cast_node, output_name


def create_reshape_layer(input_tensor_name, input_shape, output_shape, input_tensor_dtype):
    input_shape = dynamic2static(input_shape)  # fixme: dynamic shape: [batch,64]
    output_shape = dynamic2static(output_shape)
    input_size = 0 if (not len(input_shape)) or 0 in input_shape else np.prod(input_shape)
    output_size = 0 if (not len(output_shape)) or 0 in input_shape else np.prod(output_shape)
    _random_id = _get_random_id()

    # 创建一个常量张量来指定输出的目标形状
    const_shape_val = numpy_helper.from_array(np.array(output_shape, dtype=np.int64),
                                              name=f'const_shape_val_{_random_id}')
    initializers = [const_shape_val]
    concat_node = None
    splice_node = None

    flatten_output_name = f'{input_tensor_name}_flattened_{_random_id}'
    flatten_node = helper.make_node(
        'Flatten',
        inputs=[input_tensor_name],
        outputs=[flatten_output_name],
        name=f'flatten_{_random_id}',
        axis=0,  # return 2D matrix such as [1, x]
    )

    if input_size == output_size:
        output_tensor_name = input_tensor_name
        flatten_node = None

    elif input_size < output_size:
        padding_count = int(output_size - input_size)
        input_type_np = type_utils.onnx_type2numpy[input_tensor_dtype]
        padding_values = np.zeros([1, padding_count], dtype=input_type_np)
        const_pad_val = numpy_helper.from_array(padding_values, name=f'const_pad_val_{_random_id}')
        initializers.append(const_pad_val)

        output_tensor_name = f'{input_tensor_name}_{_random_id}'
        concat_node = helper.make_node(
            'Concat',
            inputs=[flatten_output_name, const_pad_val.name],
            outputs=[output_tensor_name],
            axis=1,
            name=f'concat_{_random_id}'
        )
    else:
        slice_start = [0, 0]  # notice: flatten return a 2D matrix
        slice_end = [1, output_size]
        starts_tensor = numpy_helper.from_array(np.array(slice_start, dtype=np.int64), name=f'slice_start_{_random_id}')
        ends_tensor = numpy_helper.from_array(np.array(slice_end, dtype=np.int64), name=f'slice_end_{_random_id}')
        initializers.extend([starts_tensor, ends_tensor])

        output_tensor_name = f'{input_tensor_name}_{_random_id}'
        splice_node = helper.make_node(
            'Slice',
            inputs=[flatten_output_name, starts_tensor.name, ends_tensor.name],
            outputs=[output_tensor_name],
            name=f'slice_{_random_id}'
        )

    reshape_input_name = output_tensor_name
    reshape_out_name = f"{reshape_input_name}_reshaped{_random_id}"
    reshape_node = helper.make_node(
        'Reshape',
        inputs=[reshape_input_name, const_shape_val.name],
        outputs=[reshape_out_name],
        name=f'reshape_{_random_id}'
    )
    padding_node = concat_node if concat_node else splice_node
    reshape_padding_nodes = [node for node in [flatten_node, padding_node, reshape_node] if node is not None]
    return reshape_padding_nodes, initializers, reshape_out_name


if __name__ == '__main__':
    input_tensor_name = 'input_tensor'
    input_shape = [17]
    target_shape = [2, 3]

    reshape_padding_node_list, initializers, reshape_output_name = create_reshape_layer(input_tensor_name, input_shape,
                                                                                        target_shape)
    graph = helper.make_graph(
        reshape_padding_node_list,
        'reshape_with_padding_or_splice',
        inputs=[helper.make_tensor_value_info(input_tensor_name, onnx.TensorProto.FLOAT, input_shape)],
        outputs=[
            helper.make_tensor_value_info(reshape_output_name, onnx.TensorProto.FLOAT, target_shape)],
        initializer=initializers
    )

    model = helper.make_model(graph, producer_name='pytorch')
    onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, 'reshape_model.onnx')
    # import os
    # os.system("python test_ort.py reshape_model.onnx")

    print('Reshape model with padding or splice created and saved successfully!')

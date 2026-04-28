import onnx
import onnxruntime as ort
from onnxruntime.transformers import optimizer
import numpy as np
import sys
import type_utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def random_input(model):
    inputs = {}
    for input_tensor in model.graph.input:
        input_name = input_tensor.name
        tensor_type = input_tensor.type.tensor_type
        this_dtype = type_utils.onnx_type2numpy[tensor_type.elem_type]
        if len(tensor_type.shape.dim) != 0:  # has tensor shape
            shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in tensor_type.shape.dim]
            inputs[input_name] = np.random.rand(*shape).astype(this_dtype)

        elif tensor_type.elem_type in [0, 1, 10, 11, 16, 17, 18, 19, 20]:  # const: float
            inputs[input_name] = this_dtype(np.random.random())
        elif tensor_type.elem_type in [2, 3, 4, 5, 6, 7, 13, 21, 22]:  # const: int
            inputs[input_name] = this_dtype(np.random.randint(1, 10))
        elif tensor_type.elem_type in [9]:
            inputs[input_name] = np.bool_(np.random.choice([True, False]))
        else:
            print(f"[WARNING]: Unsolved type:{tensor_type.elem_type}!")
    return inputs


if __name__ == '__main__':
    model_path = sys.argv[1]
    model = onnx.load(model_path)
    # print(f"[DEBUG]: {model_path}")
    input_data = random_input(model)

    # [Online] Run inference with the original model
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    this_provider_list = ort.get_available_providers()

    original_session = ort.InferenceSession(model_path, sess_options, providers=this_provider_list)
    output_names = [output.name for output in original_session.get_outputs()]
    original_result = original_session.run(output_names, input_data)

    # [Offline] Run inference with the optimized model
    optimized_model_path = model_path[:-5]+"_opt.onnx"
    print(optimized_model_path)
    # print(optimized_model_path)
    optimized_model = optimizer.optimize_model(model_path, opt_level=99, use_gpu=True)  # [0, 1, 2, 99]
    optimized_model.save_model_to_file(optimized_model_path)
    optimized_session = ort.InferenceSession(optimized_model_path, providers=this_provider_list)

    inputs_name_new = [input.name for input in onnx.load(optimized_model_path).graph.input]
    if inputs_name_new != input_data.keys():  # dead inputs have been deleted
        input_data = {k: input_data[k] for k in inputs_name_new}
    optimized_result = optimized_session.run(output_names, input_data)
    try:
        # 【Differential Testing】 Compare results from original and optimized models
        for r1, r2 in zip(original_result, optimized_result):
            np.testing.assert_allclose(r1, r2, atol=1e-3, rtol=1e-3)
    except Exception as e:
        if "Mismatched elements" in str(e):  # check the randomness
            original_result2 = original_session.run(output_names, input_data)
            for r1, r2 in zip(original_result, original_result2):
                is_not_random = np.allclose(r1, r2, atol=1e-3, rtol=1e-3)
                if not is_not_random:
                    assert False, str(e)
    print("[INFO] Test successfully!")

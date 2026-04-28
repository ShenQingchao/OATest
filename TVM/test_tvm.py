import subprocess
import numpy as np
import random
import tvm

test_preffix_str = '''import tvm
from tvm import relax
import numpy as np
import time
import os
'''

# use LambdaLift before vm.compiler to avoid false positive
common_pass_str = '''

mod = Module
with tvm.transform.PassContext(disabled_pass=["RemoveUnusedParameters"]):
    mod = relax.transform.FuseTIR()(mod)
mod = tvm.relax.transform.LegalizeOps()(mod)
mod = relax.transform.LambdaLift()(mod)

'''


all_funcs_defined_str = '''
def tvm_array2numpy(tvm_array):
    if isinstance(tvm_array, tvm.runtime.ndarray.NDArray):
        return tvm_array.numpy()
    if isinstance(tvm_array, tvm.ir.container.Array):
        res = [tvm_array2numpy(i) for i in tvm_array]
        return res
    return None

def compile_mod(mod, target):
    if target == 'llvm':
        with tvm.transform.PassContext(opt_level=4):
            ex = relax.build(mod, target='llvm')
        vm = relax.VirtualMachine(ex, tvm.cpu())
    else:
        with tvm.target.Target("cuda"):
            mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
        with tvm.transform.PassContext(opt_level=4):
            ex = relax.build(mod, target='cuda')
        vm = relax.VirtualMachine(ex, tvm.gpu())
    return vm

def differential_testing(before_outputs, after_outputs):
    if before_outputs is not None:
        if not isinstance(before_outputs, list):
            before_outputs = [before_outputs]
            after_outputs = [after_outputs]
        for i, output_i in enumerate(before_outputs):
            np.testing.assert_allclose(before_outputs[i], after_outputs[i], rtol=1e-3, atol=1e-3)

def execute_infer(vm, func_name, *inputs):
    try:
        before_outputs=None
        start_time = time.perf_counter()
        for i in range(1):
            mod_outputs = vm[f'{func_name}'](*inputs)
            if before_outputs:
                differential_testing(before_outputs, mod_outputs)
        end_time = time.perf_counter()
        infer_time = end_time - start_time
        mod_outputs = tvm_array2numpy(mod_outputs)
    except Exception as e:
        print(f'[Warning] Random FP or inputs error:{e}')
        os._exit(0)
    return mod_outputs, infer_time
'''


differential_testing_str = '''
differential_testing(before_outputs, after_outputs)
'''

#if infer_time2 > (infer_time1*(1+5e-2)):
# if infer_time2 > (infer_time1*3):
#     raise Exception(f"[Performance Bug]: slower {(infer_time2-infer_time1)/infer_time1}x after the optimization.")


def _gen_scalar_value(input_dtype):
    if input_dtype[:5] == "float":
        return f"{random.random()}"
    elif input_dtype[:3] == 'int' or input_dtype[:4] == 'uint':
        return f"{random.randint(1, 20)}"
    elif input_dtype == 'bool':
        return f"{random.choice([True, False])}"


def _gen_scalar(input_dtype):
    if input_dtype[:5] == "float":
        return f"np.{input_dtype}({random.random()})"
    elif input_dtype[:3] == 'int' or input_dtype[:4] == 'uint':
        return f"np.{input_dtype}({random.randint(1, 20)})"
    elif input_dtype == 'bool':
        return f"np.bool_({random.choice([True, False])})"


def _gen_input_data_str(input_shape, input_dtype):
    if input_shape is None:  # scalar without dtype
        return _gen_scalar_value(input_dtype)
    if len(input_shape) == 0:  # scalar: R.Tensor((), dtype='')
        return f"tvm.nd.array({_gen_scalar(input_dtype)})"

    if isinstance(input_shape, tuple):  # shape: ([2, 4, 26, 26], [2, 4, 26, 26])  # todo:CheckMe!
        tuple_data_str = ""
        for item in input_shape:
            item_str = _gen_input_data_str(item, input_dtype)
            tuple_data_str = tuple_data_str + item_str + ","
        # import pdb;pdb.set_trace()
        return "("+tuple_data_str + ")"

    def _change_shape2static(shape):
        new_shape = []
        for item in shape:
            if isinstance(item, str):
                new_shape.append(1)
            else:
                new_shape.append(item)
        return new_shape
    input_shape = _change_shape2static(input_shape)
    input_shape = [i for i in input_shape]  # dead code?

    input_data_str = ''
    if input_dtype[:5] == "float":
        input_data_str = f"10 * np.random.random({input_shape}).astype('{input_dtype}')"
    elif input_dtype[:3] == 'int' or input_dtype[:4] == 'uint':
        input_data_str = f"np.random.randint(10, size={input_shape}).astype('{input_dtype}')"
    elif input_dtype == 'bool':
        input_data_str = f"np.random.randint(2, size={input_shape}).astype('{input_dtype}')"
    else:
        print(f"[ERROR]: Unidentified type {input_dtype}!")

    if "np." in input_data_str:
        input_data_str = f'tvm.nd.array({input_data_str})'
    else:
        input_data_str = f'{input_data_str}'
    return input_data_str


def gen_tvm_test_file(irs, passes, irs_path, model_inputs_list, mod_input_func_name):
    new_test = test_preffix_str + irs + common_pass_str + all_funcs_defined_str

    # gen model inputs and execution
    input_name_list = []
    all_inputs_statements = ''
    if model_inputs_list:  # model_inputs_list=False: cannot get the valid input shape
        for i, input_i in enumerate(model_inputs_list):
            this_input_name = f"input_{i}"
            input_name_list.append(this_input_name)

            _, input_shape, input_dtype = input_i
            input_data_str = _gen_input_data_str(input_shape, input_dtype)
            all_inputs_statements += f"{this_input_name} = {input_data_str}\n"
            # if "np." in input_data_np_str:
            #     all_inputs_statements += f'{this_input_name} = tvm.nd.array({input_data_np_str})\n'
            # else:
            #     all_inputs_statements += f'{this_input_name} = {input_data_np_str}\n'
    new_test += all_inputs_statements
    # 1. no optimization
    new_test += f"vm1 = compile_mod(mod, 'llvm')\n"
    # 2. optimization with sequential
    passes_list_str = []
    indent_str = ""
    for _pass in passes:
        if not hasattr(tvm.relax.transform, _pass):
            continue

        required_para_pass = ['FuseOpsByPattern', 'UpdateParamStructInfo', 'AlterOpImpl',
                              'BindSymbolicVars', 'BindParams']
        if _pass in required_para_pass:  # todo: support pass with required paras
            continue
        elif _pass == 'ConvertLayout':
            passes_list_str.append(f'relax.transform.{_pass}'+'({"relax.nn.conv2d": ["NHWC", "OHWI"]})')
        elif _pass == 'UpdateVDevice':
            passes_list_str.append(f'relax.transform.{_pass}'+'(tvm.ir.VDevice("llvm"),1)')
        elif _pass == 'MetaScheduleTuneIRMod':
            passes_list_str.append(f'relax.transform.{_pass}'+'({},\"./\",1)')  # with required params
            new_test += "target = tvm.target.Target('llvm --num-cores=16')\n"
            new_test += "with target, tvm.transform.PassContext(opt_level=0):\n"
            indent_str = "    "
        elif _pass == 'MetaScheduleTuneTIR':
            passes_list_str.append(f'relax.transform.{_pass}'+'(\"./\",1)')  # with required params
            new_test += "target = tvm.target.Target('llvm --num-cores=16')\n"
            new_test += "with target, tvm.transform.PassContext(opt_level=0):\n"
            indent_str = "    "
        elif _pass == 'FewShotTuning':
            passes_list_str.append(f'relax.transform.{_pass}(1)')  # with required params
            new_test += "target = tvm.target.Target('llvm --num-cores=16')\n"
            new_test += "with target, tvm.transform.PassContext(opt_level=0):\n"
            indent_str = "    "
        elif _pass == 'Gradient':  # with required params
            passes_list_str.append(f'relax.transform.{_pass}("{mod_input_func_name}")')
        elif 'MetaScheduleApplyDatabase' in _pass:  # todo: need add environment 'with db, target:'
            continue
        elif 'sequential' in _pass:
            continue
        else:
            passes_list_str.append(f"relax.transform.{_pass}()")
    if len(passes_list_str) != 0:
        new_test += f"{indent_str}mod_seq = tvm.transform.Sequential({passes_list_str})(mod)\n".replace("'", '')
        # 3. optimization with passes one by one
        for this_pass in passes_list_str:
            new_test += f"{indent_str}mod = {this_pass}(mod)\n"
        # oracle 1: transform.Sequential([xx]) == transform.xx
        new_test += "tvm.ir.assert_structural_equal(mod_seq, mod)\n"

    # oracle 2&3: differential_testing between compiled mods with different optimizations and target devices\
    # todo: this pass will change the weights/inputs/output! inputs should be updated!!
    changed_output_num_passes = ['LazySetOutput', 'LazyGetInput',
                                 'RemoveUnusedOutputs', 'RemoveUnusedParameters',
                                 'LiftTransformParams']
    new_test += f"vm2 = compile_mod(mod, 'llvm')\n"

    if not set(passes).intersection(set(changed_output_num_passes)) and "alloc_tensor" not in irs:
        # input_name_list_cuda = [f"tvm.nd.array({i}, tvm.cuda())" for i in input_name_list]  # todo: solve OOM
        # input_name_list_cuda = input_name_list
        new_test += f"before_outputs, infer_time1 = execute_infer(vm1, '{mod_input_func_name}',{','.join(input_name_list)})\n"
        new_test += f"after_outputs, infer_time2 = execute_infer(vm2,'{mod_input_func_name}',{','.join(input_name_list)})\n"
        new_test += differential_testing_str

    new_test_path = irs_path.replace('.py', '_test.py')
    with open(new_test_path, 'w') as f:
        f.write(new_test)
    return new_test_path


def run_test(test_case_path):
    process = subprocess.Popen(
        ["python", test_case_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


import os
import tempfile
def execute_test_here(new_test_path):
    command = f"python {new_test_path}"
    with tempfile.NamedTemporaryFile(delete=True) as temp_err:
        return_code = os.system(f"{command}  2> {temp_err.name}")
        with open(temp_err.name, 'r') as f:
            stderr = f.read()
        return return_code, "", stderr


if __name__ == '__main__':
    res = _gen_input_data_str([], 'float16')
    print(res)
    print("yes!!!!!!!!!")

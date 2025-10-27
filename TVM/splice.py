import random
import numpy as np
import re
import parse_utils
import parse_shape
import parse_return
import get_func_block
import parse_all_vars
import transform_decompile

import tvm  # not dead code


def _is_dynamic_shape(shape):
    if isinstance(shape, tuple):
        res_flag = 0
        for sub_shape in shape:
            res_flag += _is_dynamic_shape(sub_shape)
        if res_flag > 0:
            return True
        return False
    if any(isinstance(item, str) for item in shape):
        return True
    return False


def _correct_var_name(var_name):
    if '[' in var_name:
        var_name = re.sub(r'\[([0-9]+)\]', r'_\1', var_name)
    return var_name


def reshape_for_connect(ori_shape, target_shape, ori_var, target_var, var_dtype, indent=''):
    reshape_connection_str = ""
    if ori_shape == target_shape:
        reshape_connection_str = f"{indent}{target_var} = {ori_var}\n"
    elif _is_dynamic_shape(ori_shape) or _is_dynamic_shape(target_shape):
        ndim_ori = len(ori_shape)
        ndim_target = len(target_shape)
        if ndim_ori == ndim_target:
            reshape_connection_str = f"{indent}{target_var} = {ori_var}\n"
        else:
            if not _is_dynamic_shape(ori_shape):
                target_reshape_val = parse_shape.adjust_static_ndims(ori_shape, ndim_target)
                reshape_connection_str += f"{indent}{target_var} = R.reshape({ori_var}, {target_reshape_val})\n"
                # import pdb;pdb.set_trace()
            else:  # para is dynamic shape
                target_reshape_val = parse_shape.adjust_dynamic_ndims(ori_shape, ndim_target)
                reshape_connection_str += f"{indent}{target_var} = R.reshape({ori_var}, {target_reshape_val})\n"
                reshape_connection_str = reshape_connection_str.replace("'", '')   # 'm' -> m
    elif isinstance(target_shape, tuple):
        seed_output_tensor_ele_num = np.prod(ori_shape) if len(ori_shape) != 0 else 0
        reshape_connection_str += f"{indent}{target_var} = list()\n"
        for sub_shape in target_shape:
            if isinstance(sub_shape, tuple):
                return ""
            this_sub_para_ele_num = np.prod(sub_shape)
            if seed_output_tensor_ele_num == this_sub_para_ele_num:
                reshape_connection_str += f"{indent}{target_var}.append(R.reshape({ori_var}, {sub_shape}))\n"
            elif this_sub_para_ele_num == 0:
                reshape_connection_str += f"{indent}{target_var} = R.zeros(R.shape([]), dtype='{var_dtype}')\n"
            elif seed_output_tensor_ele_num > this_sub_para_ele_num:
                reshape_connection_str += f"{indent}tensor_1dim = R.reshape({ori_var}, [{seed_output_tensor_ele_num}])\n"
                reshape_connection_str += f"{indent}temp = R.strided_slice(tensor_1dim, begin=[0], end=[{this_sub_para_ele_num}], strides=[1], axes=[0])\n"
                reshape_connection_str += f"{indent}{target_var}.append(R.reshape(temp, {sub_shape}))\n"
            else:
                pad_ele_num = this_sub_para_ele_num - seed_output_tensor_ele_num
                reshape_connection_str += f"{indent}tensor_1dim = R.reshape({ori_var}, [{seed_output_tensor_ele_num}])\n"
                reshape_connection_str += f"{indent}pad_tensor = R.zeros(R.shape([{pad_ele_num}]), dtype='{var_dtype}')\n"
                reshape_connection_str += f"{indent}temp = R.concat((tensor_1dim, pad_tensor), axis=-1)\n"
                reshape_connection_str += f"{indent}{target_var}.append((temp, {sub_shape}))\n"
        reshape_connection_str += f"{indent}{target_var} = tuple({target_var})\n"

    else:
        seed_output_tensor_ele_num = np.prod(ori_shape) if len(ori_shape) != 0 else 0
        this_para_ele_num = np.prod(target_shape) if len(target_shape) != 0 else 0

        if seed_output_tensor_ele_num == this_para_ele_num:
            reshape_connection_str += f"{indent}{target_var} = R.reshape({ori_var}, {target_shape})\n"
        elif this_para_ele_num == 0:
            reshape_connection_str += f"{indent}{target_var} = R.zeros(R.shape([]), dtype='{var_dtype}')\n"
        elif seed_output_tensor_ele_num > this_para_ele_num:
            reshape_connection_str += f"{indent}tensor_1dim = R.reshape({ori_var}, [{seed_output_tensor_ele_num}])\n"
            reshape_connection_str += f"{indent}temp = R.strided_slice(tensor_1dim, begin=[0], end=[{this_para_ele_num}], strides=[1], axes=[0])\n"
            reshape_connection_str += f"{indent}{target_var} = R.reshape(temp, {target_shape})\n"
        else:
            pad_ele_num = this_para_ele_num - seed_output_tensor_ele_num
            if seed_output_tensor_ele_num != 0:
                reshape_connection_str += f"{indent}tensor_1dim = R.reshape({ori_var}, [{seed_output_tensor_ele_num}])\n"
                reshape_connection_str += f"{indent}pad_tensor = R.zeros(R.shape([{pad_ele_num}]), dtype='{var_dtype}')\n"
                reshape_connection_str += f"{indent}temp = R.concat((tensor_1dim, pad_tensor), axis=-1)\n"
                reshape_connection_str += f"{indent}{target_var} = R.reshape(temp, {target_shape})\n"
            else:
                reshape_connection_str += f"{indent}temp = R.zeros(R.shape([{pad_ele_num}]), dtype='{var_dtype}')\n"
                reshape_connection_str += f"{indent}{target_var} = R.reshape(temp, {target_shape})\n"


    return reshape_connection_str


def insert_subgraph(seed, donor):
    # print(f"[Seed IR]:\n{seed}\n{'*'*66}")
    # print(f"[Donor IR]:\n{donor}\n{'*'*66}")
    try:
        updated_donor = parse_utils.update_func_name(seed, donor)
    except Exception as e:
        print("update func name fail", e)
        return None
    # print(f'[Updated donor]:\n{updated_donor}\n***************')

    # add the function call to create a new relation
    seed_subgraph_func_list = parse_utils.get_all_func_kind(irs=seed)['relax_func']
    donor_subgraph_func_list = parse_utils.get_all_func_kind(irs=updated_donor)['relax_func']

    if len(seed_subgraph_func_list) == 0 or len(donor_subgraph_func_list) == 0:
        return None  # generating failed

    seed_connection_func_name = random.choice(seed_subgraph_func_list)
    donor_connection_func_name = random.choice(donor_subgraph_func_list)
    print(f"[INFO]: seed func: '{seed_connection_func_name}'; donor func:'{donor_connection_func_name}'")
    seed_connection_func = seed[seed_connection_func_name]
    donor_connection_func = updated_donor[donor_connection_func_name]

    connection_str = ''
    # set the indent length for the connection position
    if '\n    with R.dataflow()' in seed_connection_func.script():  # dismiss inner-fuc-dataflow
        indent = '            '
    else:
        indent = '        '

    return_res = parse_return.get_return_name_constraint(seed_connection_func)
    if not return_res:
        return None
    seed_ret_var_name, seed_output_constraint, padding_str = return_res
    # if padding_str:
    #     connection_str += f"{indent}{padding_str}"  # R.TupleGetItem(gv, 0) # fixme

    seed_ret_shape_list = parse_utils.tvm_shape2list(seed_output_constraint.shape)
    print('seed_ret_shape_list:', seed_ret_shape_list)
    if _is_dynamic_shape(seed_ret_shape_list):  # dynamic shape [m, n]
        print(f"[Info]: dynamic shape:{seed_ret_shape_list}")
    else:
        seed_output_tensor_ele_num = np.prod(seed_ret_shape_list)

    # donor func constraint
    donor_func_input_constraints = parse_utils.get_func_inputs_constraints(donor_connection_func)
    if not donor_func_input_constraints:
        return None

    donor_block_body_str = get_func_block.extract_function_body(donor_connection_func.script(), indent)
    seed_used_var_dict, all_pure_var_name_type = parse_all_vars.collect_all_vars(seed_connection_func)
    print('[INFO] seed_used_var_dict:', seed_used_var_dict)

    import parse_op_constraints
    donor_inputs_types = parse_op_constraints.get_donor_inputs_types(donor_connection_func, donor_func_input_constraints)
    print('[INFO] donor_inputs_types:', donor_inputs_types)
    if donor_inputs_types == 0:  # "any" shape
        # seed_var_name = random.choice(all_pure_var_name) if all_pure_var_name else seed_ret_var_name
        # for donor_para_name, _, _ in donor_func_input_constraints:
        #     connection_str += f"{indent}{donor_para_name} = {seed_var_name}\n"
        if all_pure_var_name_type:
            seed_var_name, seed_var_dtype = random.choice(all_pure_var_name_type)
        else:
            seed_var_name, seed_var_dtype = seed_ret_var_name, None
        for donor_para_name, _, donor_para_dtype in donor_func_input_constraints:
            if seed_var_dtype and donor_para_dtype != seed_var_dtype:
                connection_str = f"{indent}{donor_para_name} = R.astype({seed_var_name}, dtype='{donor_para_dtype}')\n"
            else:
                connection_str += f"{indent}{donor_para_name} = {seed_var_name}\n"
    elif donor_inputs_types == 1:  # "same"
        if len(seed_used_var_dict) != 0 and seed_used_var_dict[0].get('list'):
            ndim = donor_func_input_constraints[0][1]
            if ndim in list(seed_used_var_dict[0]['list'].keys()):
                ndim_dict = seed_used_var_dict[0]['list'][ndim]
                seed_var_shape = random.choice(list(ndim_dict.keys()))  # keep all inputs with same shape
            else:
                ndim = random.choice(list(seed_used_var_dict[0]['list'].keys()))
                ndim_dict = seed_used_var_dict[0]['list'][ndim]
                seed_var_shape = random.choice(list(ndim_dict.keys()))
            seed_var_name, seed_var_dtype = random.choice(ndim_dict[seed_var_shape])
        else:
            seed_var_name = seed_ret_var_name
        for donor_para_name, _, donor_para_dtype in donor_func_input_constraints:
            connection_str += f"{indent}{donor_para_name} = {seed_var_name}\n"
    else:
        for donor_para_name, donor_para_shape, donor_para_dtype in donor_func_input_constraints:
            print(f"Var '{donor_para_name}' shape in donor:", donor_para_shape)
            if not donor_para_shape:  # todo: support this special case
                return None
            if seed_used_var_dict and seed_used_var_dict[0].get('list'):
                if len(donor_para_shape) in seed_used_var_dict[0]['list'].keys():
                    ndim_dict = seed_used_var_dict[0]['list'][len(donor_para_shape)]
                    if str(donor_para_shape) in ndim_dict.keys():
                        seed_var_name, seed_var_dtype = random.choice(ndim_dict[str(donor_para_shape)])
                        if donor_para_dtype != seed_var_dtype:
                            retype_str_tmp = f"R.astype({seed_var_name}, dtype='{donor_para_dtype}')\n"
                            seed_var_name = _correct_var_name(seed_var_name)
                            connection_str += f"{indent}{seed_var_name} = {retype_str_tmp}"
                        connection_str += f"{indent}{donor_para_name} = {seed_var_name}\n"
                    else:
                        seed_var_shape = random.choice(list(ndim_dict.keys()))
                        seed_var_name, seed_var_dtype = random.choice(ndim_dict[seed_var_shape])
                        seed_var_shape = eval(seed_var_shape)
                        if donor_para_dtype != seed_var_dtype:
                            retype_str_tmp = f"R.astype({seed_var_name}, dtype='{donor_para_dtype}')\n"
                            seed_var_name = _correct_var_name(seed_var_name)
                            connection_str += f"{indent}{seed_var_name} = {retype_str_tmp}"
                        # seed_var_shape --> donor_para_shape
                        reshape_str = reshape_for_connect(seed_var_shape, donor_para_shape, seed_var_name, donor_para_name,
                                                          donor_para_dtype, indent)
                        connection_str += reshape_str
                else:
                    ndim = random.choice(list(seed_used_var_dict[0]['list'].keys()))
                    ndim_dict = seed_used_var_dict[0]['list'][ndim]

                    seed_var_shape = random.choice(list(ndim_dict.keys()))
                    seed_var_name, seed_var_dtype = random.choice(ndim_dict[seed_var_shape])
                    seed_var_shape = eval(seed_var_shape)
                    if donor_para_dtype != seed_var_dtype:
                        retype_str_tmp = f"R.astype({seed_var_name}, dtype='{donor_para_dtype}')\n"
                        seed_var_name = _correct_var_name(seed_var_name)
                        seed_var_name = _correct_var_name(seed_var_name)
                        connection_str += f"{indent}{seed_var_name} = {retype_str_tmp}"
                    # 　seed_var_shape vs donor_para_shape
                    reshape_str = reshape_for_connect(seed_var_shape, donor_para_shape, seed_var_name, donor_para_name,
                                                      donor_para_dtype, indent)
                    connection_str += reshape_str
                    if isinstance(donor_para_shape, tuple):
                        connection_str += ''

            else:
                connection_str += f"{indent}{donor_para_name} = {seed_ret_var_name}\n"

    connection_str += donor_block_body_str+"\n"

    def _get_func_output_name(output_var):
        if isinstance(output_var, tvm.relax.expr.Tuple):
            if len(output_var) != 0:
                output_name = _get_func_output_name(output_var[-1])
            else:
                output_name = "R.tuple()"
        elif hasattr(output_var, 'name_hint'):
            output_name = output_var.name_hint
        else:
            output_name = str(output_var)
        return output_name

    donor_output_var = donor_connection_func.body.body
    donor_output_name = _get_func_output_name(donor_output_var)

    if '\n    with R.dataflow()' in seed_connection_func.script():
        if donor_output_name != "R.tuple()":
            connection_str += f"{indent}R.output({donor_output_name})\n"
    connection_str += f"        return {donor_output_name}\n"

    for _donor_func_name, func_body in updated_donor.functions.items():
        seed[_donor_func_name] = func_body

    # connection
    seed_irs_str = seed.script(show_meta=True)
    seed_irs_str_list = seed_irs_str.split("\n    @R.function")
    new_ir_str = seed_irs_str_list[0]
    # match the specific connection function rather than all place
    for sub_str in seed_irs_str_list[1:]:
        if f"def {seed_connection_func_name}(" not in sub_str:
            new_ir_str += f"\n    @R.function{sub_str}"
        else:  # connected func point
            # change private=False
            sub_str = sub_str.replace('(private=True)', '(private=False)')

            connection_pos1 = re.compile(rf"^{indent}R\.output.*?\n        return.*?$", flags=re.MULTILINE)
            connection_pos2 = re.compile(rf"^        return.*?$", flags=re.MULTILINE)
            if connection_pos1.search(sub_str):
                new_sub_irs = re.sub(connection_pos1, connection_str, sub_str)
                new_ir_str += f"\n    @R.function{new_sub_irs}"
            elif connection_pos2.search(sub_str):
                connection_str = connection_str.replace(f"{indent}R.output(res)\n", "")   # R.output only in dataflow
                new_sub_irs = re.sub(connection_pos2, connection_str, sub_str)
                new_ir_str += f"\n    @R.function{new_sub_irs}"
            else:
                print(f"[Error]: Synthesize failed! Cannot find valid connection place!")
                return None
    # add metadata
    if 'metadata[' in donor_connection_func.script():
        donor_irs_str = donor.script(show_meta=True)
        donor_metadata_str = donor_irs_str.split('}""")\n')[0]+'}""")\n'
        if 'metadata[' not in seed_irs_str:
            new_ir_str = donor_metadata_str + new_ir_str
        else:
            # combine two metadata
            seed_metadata_str = seed_irs_str.split('}""")\n')[0] + '}""")\n'
            if seed_metadata_str != donor_metadata_str:
                new_ir_str = new_ir_str  # todo: combine two compiled metadata in compact_json into one
    try:
        if "R.assert_op" in new_ir_str:  # remove the assert_op in IR
            new_ir_str = re.sub(r'^\s*R\.assert_op.*$\n', '', new_ir_str, flags=re.MULTILINE)
        print(new_ir_str)
        new_ir_mod = tvm.script.from_source(new_ir_str)   # check the validity of synthesized IRs
        new_ir_mod = tvm.relax.transform.LegalizeOps()(new_ir_mod)  # correct the func return para shape
        new_ir_str = new_ir_mod.script(show_meta=True)
        print(f'[Synthesized IR]:\n')
        new_ir_mod.show()
        model_inputs_list = parse_utils.get_func_inputs_constraints(new_ir_mod[seed_connection_func_name])
        print(f"Model inputs:\n {model_inputs_list}")
        print('*'*66)
    except Exception as e:
        print(f"[ERROR]: cannot load the synthesized IR from string!\n{e}")
        # with open("invalid_irs.py", 'a') as f:
        #     f.write(f"donor_connection_func: \n{donor_connection_func.script()}\n")
        #     f.write(new_ir_str)
        #     f.write(f"\n{'='*50}\n")
        return False  # return False: gen a invalid test; return None: gen nothing
    return [new_ir_str, model_inputs_list, seed_connection_func_name]


if __name__ == '__main__':
    import tvm
    donor_irs = '''# from tvm.script import relax as R

@R.function
def main_8(data: R.Tensor((1, 64, 56, 56), dtype="float32"), weight: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
    with R.dataflow():
        conv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(data, weight, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
        relu1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(conv1)
        gelu1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.gelu(conv1)
        out: R.Tensor((1, 64, 54, 54), dtype="float32") = R.add(relu1, gelu1)
        R.output(out)
    return out

    '''

    base_irs = '''# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main_7(x: R.Tensor(("batch_size", 1024), dtype="float16"), w1: R.Tensor((1024, 1024), dtype="float16"), w2: R.Tensor((1024, "M"), dtype="float16")) -> R.Tuple(R.Tensor(("batch_size", 1024), dtype="float16"), R.Tensor(("batch_size", "M"), dtype="float16")):
        batch_size = T.int64()
        M = T.int64()
        with R.dataflow():
            matmul1: R.Tensor((batch_size, 1024), dtype="float16") = R.matmul(x, w1, out_dtype="void")
            matmul2: R.Tensor((batch_size, M), dtype="float16") = R.matmul(x, w2, out_dtype="void")
            out: R.Tuple(R.Tensor((batch_size, 1024), dtype="float16"), R.Tensor((batch_size, M), dtype="float16")) = matmul1, matmul2
            R.output(out)
        return out

    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        R.func_attr({"relax.force_pure": 1})
        alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
        alloc = R.astype(alloc, dtype='float16')
        x = alloc
        alloc = R.astype(alloc, dtype='float16')
        tensor_1dim = R.reshape(alloc, [6])
        pad_tensor = R.zeros(R.shape([1048570]), dtype='float16')
        temp = R.concat((tensor_1dim, pad_tensor), axis=-1)
        w1 = R.reshape(temp, [1024, 1024])
        alloc = R.astype(alloc, dtype='float16')
        w2 = alloc
        batch_size = T.int64()
        M = T.int64()
        matmul1=R.matmul(x, w1, out_dtype="void")
        matmul2=R.matmul(x, w2, out_dtype="void")
        out=matmul1, matmul2
        return out

'''


    base_irs = tvm.script.from_source(base_irs)
    print('*' * 1000)
    base_irs.show()

    donor_irs = tvm.script.from_source(donor_irs)
    donor_irs.show()

    temp = insert_subgraph(base_irs, donor_irs)
    if temp:
        a,b,c = temp
        print(b)
        print("finish ALL!")


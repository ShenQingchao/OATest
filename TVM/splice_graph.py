import random
import numpy as np
import re
import parse_utils
import parse_shape
import parse_return
import tvm  # not dead code


def _is_dynamic_shape(shape):
    if any(isinstance(item, str) for item in shape):
        return True


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
    para_list_str = ''
    # set the indent length for the connection position
    if '\n    with R.dataflow()' in seed_connection_func.script():  # dismiss inner-fuc-dataflow
        indent = '            '
    else:
        indent = '        '

    return_res = parse_return.get_return_name_constraint(seed_connection_func)
    if not return_res:
        return None
    seed_ret_var_name, seed_output_constraint, padding_str = return_res
    if padding_str:
        connection_str += f"{indent}{padding_str}"

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

    for cnt, donor_para in enumerate(donor_func_input_constraints):
        para_list_str += f'para{cnt},'
        donor_para_shape = parse_utils._get_para_shape(donor_para)
        print("This_shape_from_donor:", donor_para_shape)
        if not donor_para_shape:  # todo: support this special case
            return None

        # Retype
        donor_para_dtype = donor_para.dtype if (hasattr(donor_para, 'dtype') and donor_para.dtype != '') else "float32"
        seed_output_dtype = seed_output_constraint.dtype if seed_output_constraint.dtype != '' else "float32"
        if donor_para_dtype != seed_output_constraint.dtype:
            connection_str += f"{indent}{seed_ret_var_name} = R.astype({seed_ret_var_name}, dtype='{donor_para_dtype}')\n"

        # Reshape
        if donor_para_shape == seed_ret_shape_list:
            connection_str += f"{indent}para{cnt} = {seed_ret_var_name}\n"
        # Reshape --> dynamic shape
        elif _is_dynamic_shape(donor_para_shape) or _is_dynamic_shape(seed_ret_shape_list):
            # constraints: keep same ndim & assign the same value for same dynamic shape (e.g., [m,m])
            ndim_donor = len(donor_para_shape)
            ndim_seed = len(seed_ret_shape_list)
            if ndim_donor == ndim_seed:
                connection_str += f"{indent}para{cnt} = {seed_ret_var_name}\n"
            else:
                if not _is_dynamic_shape(seed_ret_shape_list):  # seed return value is static shape
                    # constraints, target_reshape_val = parse_shape.adjust_static_ndims(seed_ret_shape_list, donor_para_shape)
                    # if constraints:
                    #     print(f"[Error]: cannot resolve this constraints!")
                    #     return None  # todo: support it
                    target_reshape_val = parse_shape.adjust_static_ndims(seed_ret_shape_list, ndim_donor)
                else:  # para is dynamic shape
                    target_reshape_val = parse_shape.adjust_dynamic_ndims(seed_ret_shape_list, ndim_donor)
                connection_str += f"{indent}para{cnt} = R.reshape({seed_ret_var_name}, {target_reshape_val})\n"
        else:  # Reshape --> static shape
            this_para_ele_num = np.prod(donor_para_shape)
            if seed_output_tensor_ele_num == this_para_ele_num:
                connection_str += f"{indent}para{cnt} = R.reshape({seed_ret_var_name}, {donor_para_shape})\n"
            elif this_para_ele_num == 0:
                connection_str += f"{indent}para{cnt} = R.zeros(R.shape([]), dtype='{seed_output_dtype}')\n"
            elif seed_output_tensor_ele_num > this_para_ele_num:
                connection_str += f"{indent}tensor_1dim = R.reshape({seed_ret_var_name}, [{seed_output_tensor_ele_num}])\n"
                connection_str += f"{indent}temp = R.strided_slice(tensor_1dim, begin=[0], end=[{this_para_ele_num}], strides=[1], axes=[0])\n"
                connection_str += f"{indent}para{cnt} = R.reshape(temp, {donor_para_shape})\n"
            else:
                pad_ele_num = this_para_ele_num - seed_output_tensor_ele_num
                connection_str += f"{indent}tensor_1dim = R.reshape({seed_ret_var_name}, [{seed_output_tensor_ele_num}])\n"
                connection_str += f"{indent}pad_tensor = R.zeros(R.shape([{pad_ele_num}]), dtype='{donor_para_dtype}')\n"
                connection_str += f"{indent}temp = R.concat((tensor_1dim, pad_tensor), axis=-1)\n"
                connection_str += f"{indent}para{cnt} = R.reshape(temp, {donor_para_shape})\n"

    connection_str += f"{indent}res = Module.{donor_connection_func_name}({para_list_str[:-1]})\n"

    # connection position: f"R.output({seed_ret_var_name})"
    connection_str += f"{indent}R.output(res)\n"
    connection_str += f"        return res\n"

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
    donor_irs = '''# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,),dtype="float16"), A: R.Tensor((32, 2),dtype="float16"), B: R.Tensor((2, 16),dtype="float16")) -> R.Tensor((32,)):
        R.func_attr({"num_input": 1})
        weight: R.Tensor((32, 16)) = R.matmul(A, B, out_dtype="void")
        out: R.Tensor((32,)) = R.matmul(weight, x, out_dtype="void")
        return out
    '''

    base_irs = '''# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main_1(x: R.Tensor(("batch_size", 1024), dtype="float16"), w1: R.Tensor((1024, 1024), dtype="float16"), w2: R.Tensor((1024, "M"), dtype="float16")) -> R.Tuple(R.Tensor(("batch_size", 1024), dtype="float16"), R.Tensor(("batch_size", "M"), dtype="float16")):
        batch_size = T.int64()
        M = T.int64()
        with R.dataflow():
            matmul1: R.Tensor((batch_size, 1024), dtype="float16") = R.matmul(x, w1, out_dtype="void")
            matmul2: R.Tensor((batch_size, M), dtype="float16") = R.matmul(x, w2, out_dtype="void")
            out: R.Tuple(R.Tensor((batch_size, 1024), dtype="float16"), R.Tensor((batch_size, M), dtype="float16")) = matmul1, matmul2
            R.output(out)
        return out

    @R.function
    def main(a: R.Tensor((16,), dtype="float32"), b: R.Tensor((16,), dtype="float32"), c: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
        R.func_attr({"num_input": 1})
        expr: R.Tensor((16,), dtype="float32") = a
        expr_1: R.Tensor((16,), dtype="float32") = R.add(expr, b)
        expr_2: R.Tensor((16,), dtype="float32") = R.add(expr_1, c)
        expr_2 = R.astype(expr_2, dtype='float16')
        x = R.reshape(expr_2, [16, 1])
        expr_2 = R.astype(expr_2, dtype='float16')
        tensor_1dim = R.reshape(expr_2, [16])
        pad_tensor = R.zeros(R.shape([1048560]), dtype='float16')
        temp = R.concat((tensor_1dim, pad_tensor), axis=-1)
        w1 = R.reshape(temp, [1024, 1024])
        expr_2 = R.astype(expr_2, dtype='float16')
        w2 = R.reshape(expr_2, [16, 1])
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


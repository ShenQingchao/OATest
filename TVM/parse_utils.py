import tvm
import random
import numpy as np
import re


def get_all_func_kind(irs):
    all_func_dict = {'relax_func': [], 'tir_func': []}
    for func_name, func_body in irs.functions.items():
        # func_name = func_name.astext(show_meta_data=False).split('\n')[1][1:]
        func_name = func_name.name_hint
        if isinstance(func_body, tvm.relax.expr.Function):
            all_func_dict['relax_func'].append(func_name)
        elif isinstance(func_body, tvm.tir.function.PrimFunc):
            all_func_dict['tir_func'].append(func_name)
        else:
            print(f"[WARNING]: Wrong func type for the function: {func_body}")
    return all_func_dict


def update_irs_inner_fun_name(irs):
    def _get_all_inner_func_name(irs):
        all_inner_func_name = set()
        for func_body in irs.functions.values():
            if isinstance(func_body, tvm.tir.function.PrimFunc):
                continue
            if isinstance(func_body, tvm.relax.expr.Function):
                for block in func_body.body.blocks:
                    bindings = block.bindings
                    for bind in bindings:
                        call = bind.value
                        if isinstance(call, tvm.relax.expr.Function):
                            all_inner_func_name.add(bind.var.name_hint)
        return all_inner_func_name

    all_inner_func_name = _get_all_inner_func_name(irs)
    irs_str = irs.script(show_meta=True)
    for old_name in all_inner_func_name:
        new_name = old_name + f"{random.randint(0,100)}"
        irs_str = irs_str.replace(f'def {old_name}(', f'def {new_name}(')
        irs_str = irs_str.replace(f' {old_name}(', f' {new_name}(')
    new_ir_str = tvm.script.from_source(irs_str)
    return new_ir_str


def update_func_name(seed, donor):
    def _update_irs_var_name(irs, todo_update_name_list):
        print('todo_update_name_list:', todo_update_name_list)
        irs_str = irs.script(show_meta=True)
        irs_str = irs_str.replace("R.call_packed", "R.call_pure_packed")
        for item in todo_update_name_list:
            old_name = item[0]
            new_name = item[1]
            irs_str = irs_str.replace(f'cls.{old_name}', f'cls.{new_name}')
            irs_str = irs_str.replace(f'def {old_name}', f'def {new_name}')
        new_ir_str = tvm.script.from_source(irs_str)
        return new_ir_str

    # change function to IRModule
    if isinstance(donor, tvm.relax.expr.Function):
        match = re.search(r"\n\s*def\s+(\w+)\s*\(", donor.script())
        func_name = match.group(1) if match else None
        donor = tvm.ir.module.IRModule({func_name: donor})

    todo_update_name_list = []
    seed_func_name_list = [item.name_hint for item in seed.functions.keys()]
    donor_func_name_list = [item.name_hint for item in donor.functions.keys()]

    for _donor_func_name_var, func_body in donor.functions.items():
        _donor_func_name = _donor_func_name_var.name_hint
        if _donor_func_name in seed_func_name_list:
            # rename the donor func_name and it's dependency (e.g., cls.XX)
            new_donor_func_name = _donor_func_name + f"_{random.randint(0, 10)}"
            # re-rename again
            if new_donor_func_name in seed_func_name_list or new_donor_func_name in donor_func_name_list:
                new_donor_func_name = _donor_func_name + f"_{random.randint(10, 32767)}"
            todo_update_name_list.append([_donor_func_name, new_donor_func_name])
    updated_donor = _update_irs_var_name(donor, todo_update_name_list)
    return updated_donor


def get_func_inputs_constraints(input_func):
    # get the model inputs with dtype and dshape
    model_inputs_constraint_list = []
    input_func_para = input_func.params
    _donor_before_para_shape = None
    for para in input_func_para:
        if isinstance(para.struct_info, tvm.relax.struct_info.TupleStructInfo):
            tuple_shape_constraint = []
            for item in para:
                if isinstance(item.struct_info, tvm.relax.struct_info.TupleStructInfo):
                    return False
                para_dtype = item.struct_info.dtype if item.struct_info.dtype != '' else "float32"
                para_dshape = _get_para_shape(item.struct_info, _donor_before_para_shape)
                _donor_before_para_shape = para_dshape
                tuple_shape_constraint.append(para_dshape)
                if para_dshape is False:
                    return False
            model_inputs_constraint_list.append([para.name_hint, tuple(tuple_shape_constraint), para_dtype])
        else:
            para_dtype = para.struct_info.dtype if hasattr(para.struct_info, 'dtype') and para.struct_info.dtype != '' else "float32"
            para_dshape = _get_para_shape(para.struct_info, _donor_before_para_shape)
            _donor_before_para_shape = para_dshape
            if para_dshape is False:
                return False
            model_inputs_constraint_list.append([para.name_hint, para_dshape, para_dtype])
    return model_inputs_constraint_list


def _get_para_shape(para_struct_info, _refer_before_para_shape=None):
    if hasattr(para_struct_info, 'shape') and para_struct_info.shape:
        para_dshape = para_struct_info.shape
        para_dshape = tvm_shape2list(para_dshape)
    elif hasattr(para_struct_info, 'values'):  # dynamic shape with [m, n]
        para_dshape = para_struct_info
        para_dshape = tvm_shape2list(para_dshape)
    elif hasattr(para_struct_info, 'ndim'):
        if _refer_before_para_shape and len(_refer_before_para_shape) == para_struct_info.ndim:
            para_dshape = _refer_before_para_shape
        else:
            # dynamic shape with ndim. e.g., R.Tensor(dtype="int64", ndim=1)
            if para_struct_info.ndim >= 0:
                para_dshape = np.random.randint(1, 5, size=para_struct_info.ndim).tolist()
            elif para_struct_info.ndim == -1:  # -1 means: unknown/random
                para_dshape = np.random.randint(1, 10, size=4).tolist()
            else:  # illegal irs, skip it!
                return False
    elif isinstance(para_struct_info, tvm.relax.struct_info.TupleStructInfo):
        # todo: the para is a tuple with multiple tensor
        # import pdb;pdb.set_trace()
        para_dshape = []
        for item in para_struct_info.fields:
            item_shape = _get_para_shape(item)
            para_dshape.append(item_shape)
        para_dshape = tuple(para_dshape)
    elif isinstance(para_struct_info, tvm.relax.struct_info.PrimStructInfo):  # R.Prim("bool")
        return None  # scalar
    else:
        return False
    return para_dshape


def tvm_shape2list(shape):
    ret_list = []
    if shape:
        for item in shape.values:
            if isinstance(item, tvm.tir.Var):
                ret_list.append(item.name)  # ['m', 'n']
            elif isinstance(item, tvm.tir.expr.IntImm):
                ret_list.append(item.value)
            else:
                ret_list.append(str(item))
    return ret_list


if __name__ == '__main__':
    from tvm.script import ir as I
    from tvm.script import relax as R

    @I.ir_module
    class Module:
        @R.function
        def fused_relax_nn_conv2d_relax_nn_relu_dnnl(data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
                                                     weight11: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor(
            (1, 64, 56, 56), dtype="float32"):
            R.func_attr({"Codegen": "dnnl"})

            # from tvm.script import relax as R

            @R.function
            def gv14(data2: R.Tensor((1, 64, 56, 56), dtype="float32"),
                     weight12: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
                R.func_attr({"Composite": "dnnl.conv2d_relu"})
                with R.dataflow():
                    lv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(data2, weight12, strides=[1, 1],
                                                                                 padding=[1, 1, 1, 1], dilation=[1, 1],
                                                                                 groups=1, data_layout="NCHW",
                                                                                 kernel_layout="OIHW",
                                                                                 out_layout="NCHW", out_dtype="void")
                    gv2: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv)
                    R.output(gv2)
                return gv2

            gv11: R.Tensor((1, 64, 56, 56), dtype="float32") = gv14(data1, weight11)
            return gv11

        @R.function
        def main(data: R.Tensor((1, 64, 56, 56), dtype="float32"),
                 weight1: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            cls = Module
            with R.dataflow():
                gv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d_relax_nn_relu_dnnl(data,
                                                                                                              weight1)
                R.output(gv)
            return gv


    res = update_func_name(Module, Module)
    print(res)

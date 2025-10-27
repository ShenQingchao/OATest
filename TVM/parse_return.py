import tvm


def _resolve_recursive_tuple(seed_ret_vars, seed_output_constraint, tuple_pos=-1):
    if isinstance(seed_ret_vars, tvm.relax.expr.Tuple):   # return (A, [B, C])
        seed_ret_vars = seed_ret_vars[tuple_pos]
        seed_output_constraint = seed_output_constraint.fields[-1]
        return _resolve_recursive_tuple(seed_ret_vars, seed_output_constraint, tuple_pos)
    else:
        return seed_ret_vars, seed_output_constraint


def get_return_name_constraint(seed_connection_func):
    padding_str = None
    # get the shape of the connected seed function in return value
    seed_output_constraint = seed_connection_func.ret_struct_info  # ret_struct_info: attrs: shape/dtype
    print('[INFO] Init seed_output_constraint:', seed_output_constraint)
    tuple_pos = 0  # [0, -1]
    if isinstance(seed_output_constraint, tvm.relax.struct_info.TupleStructInfo):
        ret_para_num = len(seed_output_constraint.fields)
        if ret_para_num == 0:  # return R.tuple()
            print(f"[Warning]: Skip it due to seed lacking return any value!")
            return False
        else:  # todo: use all para rather than choice the first one
            new_seed_output_constraint = seed_output_constraint.fields[tuple_pos]
    elif isinstance(seed_output_constraint, tvm.relax.struct_info.ObjectStructInfo):  # return tensor
        print(f"[Warning]: Skip the dynamic shape:{seed_output_constraint}")
        # todo: get the dynamic shape from this case
        return False

    # get the return var name of the seed connection func
    seed_ret_vars = seed_connection_func.body.body
    if isinstance(seed_ret_vars, tvm.relax.expr.Tuple):
        seed_ret_vars, seed_output_constraint = _resolve_recursive_tuple(seed_ret_vars, seed_output_constraint, tuple_pos)
        if isinstance(seed_ret_vars, tvm.relax.expr.Constant):  # return (D, R.prim_value(42))
            return False
        else:
            seed_ret_var_name = seed_ret_vars.name_hint

    elif isinstance(seed_output_constraint, tvm.relax.struct_info.TupleStructInfo):  # R.output() in IRs
        seed_ret_var_name = seed_ret_vars.name_hint
        seed_output_constraint = new_seed_output_constraint
        padding_str = f"{seed_ret_var_name} = R.TupleGetItem({seed_ret_var_name}, {tuple_pos})\n"
    else:
        seed_ret_var_name = seed_ret_vars.name_hint  # return A

    # check the validity
    if not hasattr(seed_output_constraint, 'shape') or not seed_output_constraint.shape:  # R.Shape(ndim=3)/R.Tensor
        print(f"[Warning]: Skip the dynamic shape with ndim:{seed_output_constraint}")
        return False
    return seed_ret_var_name, seed_output_constraint, padding_str


if __name__ == '__main__':
    base_irs = '''# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module(check_well_formed=False)
class Module:
    @T.prim_func
    def ones(A: T.Buffer((2, 3), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("T_zeros"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads()
                T.writes(A[ax0, ax1])
                A[ax0, ax1] = 0

    @R.function
    def foo(x: R.Tensor((2, 3), dtype="int32")) -> R.Tensor((2, 3), dtype="int32"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        gv0: R.Tensor((2, 3), dtype="int32") = R.call_tir_inplace(cls.ones, (x,),
                                                                  out_sinfo=R.Tensor((2, 3), dtype="int32"),
                                                                  inplace_indices=[0])
        return gv0

    '''
    base_irs = tvm.script.from_source(base_irs)
    print('*' * 1000)
    base_irs.show()
    seed_connection_func = base_irs["foo"]
    res = get_return_name_constraint(seed_connection_func)
    print(res)

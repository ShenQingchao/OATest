from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

import parse_utils, parse_return
import tvm
from tvm import relax
import re
import random

# TODO: add all op mapping
op_mapping_dict = {'ones': 'ones_like',
                     'zeros': 'zeros_like',
                     'concatenate': 'concat',
                     'pool2d': 'max_pool2d',
                     'sum': 'add',
                     'sub': 'subtract',
                     'clip': 'relu'  # fixme: a workaround
                   }


def _get_random_op(para_num):
    unary_op_list = ['tanh', 'sigmoid', 'relu', 'silu', 'softmax', ]
    binary_op_list = ['add', 'subtract', ]

    if para_num == 1:  # unary
        return random.choice(unary_op_list)
    elif para_num == 2:  # binary
        return random.choice(binary_op_list)
    else:
        return "concat"


def _get_abstract_call(true_op_name, true_op_args):
    if hasattr(R, true_op_name):
        true_op = tvm.ir.Op.get(f"relax.{true_op_name}")
    elif hasattr(R.nn, true_op_name):
        true_op = tvm.ir.Op.get(f"relax.nn.{true_op_name}")
    new_call = relax.Call(true_op, true_op_args, None, None).script()
    if true_op_name in ["concat"]:
        new_call = re.sub(rf'(R\.{true_op_name}\()([^()]+)(\))', r'\1[\2]\3', new_call)
    return new_call


def _decompile_fun(fun_ir, new_mod_str):
    # func_inputs = parse_utils.get_func_inputs_constraints(fun_ir)
    func_outputs = parse_return.get_return_name_constraint(fun_ir)
    # print('func_inputs: ', func_inputs)
    # print('func_outputs: ', func_outputs)

    for block in fun_ir.body.blocks:
        bindings = block.bindings
        for bind in bindings:
            var_name = bind.var
            call = bind.value
            call_str = call.script()
            call_str = call_str.replace("R.call_tir(", "R.call_tir(cls.")\
                .replace("R.call_tir_inplace(", "R.call_tir_inplace(cls.")

            # print(call_str)
            # print(var_name, call.op.name)

            # import pdb;pdb.set_trace()
            if isinstance(call, tvm.relax.expr.Function):  # subfunc
                # import pdb;pdb.set_trace()
                new_mod_str = _decompile_fun(call, new_mod_str)

            elif hasattr(call, "op") and hasattr(call.op, "name"):
                if call.op.name.startswith("relax.call_tir"):
                    true_op_var = call.args[0]
                    assert isinstance(true_op_var, tvm.ir.expr.GlobalVar)
                    true_op_name = true_op_var.name_hint
                    true_op_name = true_op_name.split("_inplace")[0].split("tir_")[-1]
                    true_op_name = re.sub(r"\d+$", '', true_op_name)
                    true_op_name = true_op_name[:-1] if true_op_name.endswith("_") else true_op_name
                    true_op_args = list(call.args[1])
                    # import pdb;pdb.set_trace()

                    if true_op_name in op_mapping_dict.keys():
                        true_op_name = op_mapping_dict[true_op_name]
                    # import pdb;pdb.set_trace()

                    # TODO: save the custom op and tir func
                    if not hasattr(R, true_op_name) and not hasattr(R.nn, true_op_name):
                        print(f"[Warning]: can not find corresponding op for {true_op_name}")
                        true_op_name = _get_random_op(len(true_op_args))

                    if true_op_name == 'reshape':
                        true_op_shape = parse_utils.tvm_shape2list(call.struct_info.shape)
                        new_call = f"R.reshape({true_op_args[0]}, {true_op_shape})"
                    elif true_op_name == 'strided_slice':
                        this_para_ele_num = parse_utils.tvm_shape2list(call.struct_info.shape)[0]
                        new_call = f"R.strided_slice({true_op_args[0]}, begin=[0], end=[{this_para_ele_num}], strides=[1], axes=[0])"
                    elif true_op_name == 'expand_dims':
                        new_call = f"R.expand_dims({true_op_args[0]}, axis=-1)"
                    elif true_op_name == 'split':
                        new_call = f"R.split({true_op_args[0]}, axis=-1)"
                    elif true_op_name == "layer_norm":
                        new_call = f"R.nn.layer_norm({true_op_args[0]}, {true_op_args[1]}, {true_op_args[2]}, axes=[-2, -1])"
                    else:
                        new_call = _get_abstract_call(true_op_name, true_op_args)
                    new_call = f"{var_name} = {new_call}"

                    pattern = re.escape(var_name.name_hint) + r"[:| ].*?" + re.escape(call_str)
                    new_mod_str = re.sub(pattern, new_call, new_mod_str)
                elif call.op.name == "relax.call_dps_packed":
                    true_op_args = list(call.args[1])
                    true_op_name = _get_random_op(len(true_op_args))

                    new_call = _get_abstract_call(true_op_name, true_op_args)
                    new_call = f"{var_name} = {new_call}"

                    pattern = re.escape(var_name.name_hint) + r"[:| ].*?" + re.escape(call_str)
                    new_mod_str = re.sub(pattern, new_call, new_mod_str)
                else:
                    pattern = re.escape(var_name.name_hint + ": ") + r".*?\) = "
                    new_mod_str = re.sub(pattern, var_name.name_hint + "=", new_mod_str)

            else:
                pattern = re.escape(var_name.name_hint + ": ") + r".*?\) = "
                new_mod_str = re.sub(pattern, var_name.name_hint + "=", new_mod_str)
    return new_mod_str


def decompile_ir(m):
    new_mod_str = m.script(show_meta=True)
    all_funcs = parse_utils.get_all_func_kind(m)
    for fun_name in all_funcs['relax_func']:
        fun_ir = m[fun_name]
        new_mod_str = _decompile_fun(fun_ir, new_mod_str)

    # print(new_mod_str)
    new_ir = tvm.script.from_source(new_mod_str)
    new_ir = tvm.relax.transform.DeadCodeElimination()(new_ir)
    # new_ir.show()
    return new_ir


if __name__ == '__main__':

    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
            T.func_attr({"op_pattern": 0})
            T.evaluate(0)

        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(R.Tensor((2, 3), dtype="float32"),
                                                                  R.Tensor((2, 3), dtype="float32")):
            cls = Module
            with R.dataflow():
                a = R.call_tir(cls.exp, (x,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
                b = R.call_tir(cls.exp, (a,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
                c = R.call_dps_packed("packed_dps", (a,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
                R.output(b, c)
            return (b, c)

        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")):
            cls = Module
            with R.dataflow():
                res = cls.foo(x)
                R.output(res)
            return res

    m = Module
    import pdb;pdb.set_trace()



    # @I.ir_module
    # class Module:
    #     @T.prim_func(private=True)
    #     def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
    #         T.func_attr({"op_pattern": 0})
    #         T.evaluate(0)
    #
    #     @R.function
    #     def main(x1: R.Tensor((10, 5), dtype="float32"), y1: R.Tensor((10, 5), dtype="float32")) -> R.Tensor((10, 5),
    #                                                                                                          dtype="float32"):
    #         n = T.int64()
    #         m = T.int64()
    #
    #         # from tvm.script import tir as T
    #         # from tvm.script import relax as R
    #
    #         @R.function
    #         def inner(x2: R.Tensor((n, m), dtype="float32"), y2: R.Tensor((n, m), dtype="float32")) -> R.Tensor((n, m),
    #                                                                                                             dtype="float32"):
    #             cls = Module
    #             sum_inner: R.Tensor((n, m), dtype="float32") = R.add(x2, y2)
    #             a = R.call_tir(cls.exp, (x2,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
    #             return sum_inner
    #
    #         sum_main: R.Tensor((10, 5), dtype="float32") = inner(x1, y1)
    #         return sum_main

    # @I.ir_module
    # class Module:
    #     @T.prim_func
    #     def zeros(var_T_full: T.handle):
    #         T.func_attr({"tir.noalias": T.bool(True)})
    #         n = T.int64()
    #         T_full = T.match_buffer(var_T_full, (n, n))
    #         # with T.block("root"):
    #         for ax0, ax1 in T.grid(n, n):
    #             with T.block("T_full"):
    #                 v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
    #                 T.reads()
    #                 T.writes(T_full[v_ax0, v_ax1])
    #                 T_full[v_ax0, v_ax1] = T.float32(0)
    #
    #     @R.function
    #     def main(shape: R.Shape(["n"])) -> R.Shape(["n"]):
    #         n = T.int64()
    #         R.func_attr({"num_input": 1})
    #         cls = Module
    #         with R.dataflow():
    #             zeros = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((n, n), dtype="float32"))
    #             R.output()
    #         return shape

    res = decompile_ir(Module)
    print(res)

    #
    # import parse_utils
    # # all_inner_func = parse_utils.get_all_inner_func_name(Module)
    # m = parse_utils.updata_irs_inner_fun_name(Module)
    # m.show()
    #

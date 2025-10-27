import tvm
import parse_utils


def collect_all_vars(func_ir):
    all_var_constraints = []   # [var_dict, var_dict]  each item represent a block
    all_pure_var_name = []

    # func_ir.show()
    try:
        for block in func_ir.body.blocks:
            var_dict = {}  # key : var_type -> tensor_ndim -> tensor_shape
            for binding in block.bindings:
                if hasattr(binding, 'value') and hasattr(binding.value, 'op') and hasattr(binding.value.op, 'name') \
                        and binding.value.op.name == 'relax.tensor_to_shape':
                    continue
                var_node = binding.var
                var_name = var_node.name_hint
                var_shape = parse_utils._get_para_shape(var_node.struct_info)
                if not var_shape:
                    continue
                var_dtype = var_node.struct_info.dtype if hasattr(var_node.struct_info, 'dtype') else "float32"

                var_type = type(var_shape).__name__  # Tensor, Tuple
                if var_type not in var_dict.keys():
                    var_dict[var_type] = {}
                ndim = len(var_shape) if isinstance(var_shape, list) else len(var_shape[0])
                if ndim not in var_dict[var_type]:
                    var_dict[var_type][ndim] = {}
                if str(var_shape) not in var_dict[var_type][ndim]:
                    var_dict[var_type][ndim][str(var_shape)] = []
                var_dict[var_type][ndim][str(var_shape)].append([var_name, var_dtype])
                all_pure_var_name.append([var_name, var_dtype])

                # further parse the tuple into multiple tensor var
                if isinstance(var_shape, tuple):
                    for i, sub_var_shape in enumerate(var_shape):
                        sub_var_type = type(sub_var_shape).__name__
                        if isinstance(sub_var_shape, tuple):
                            for j, sub_sub_var_shape in enumerate(sub_var_shape):
                                sub_sub_var_type = type(sub_var_shape).__name__
                                if sub_sub_var_type not in var_dict.keys():
                                    var_dict[sub_sub_var_type] = {}
                                sub_sub_ndim = len(sub_sub_var_shape)
                                if sub_sub_ndim not in var_dict[sub_sub_var_type]:
                                    var_dict[sub_sub_var_type][sub_sub_ndim] = {}
                                if str(sub_sub_var_shape) not in var_dict[sub_sub_var_type][sub_sub_ndim]:
                                    var_dict[sub_sub_var_type][sub_sub_ndim][str(sub_sub_var_shape)] = []
                                    var_dict[sub_sub_var_type][sub_sub_ndim][str(sub_sub_var_shape)].append(
                                        [var_name + f"[{i}][{j}]", var_node[i][j].struct_info.dtype])
                                    all_pure_var_name.append([var_name + f"[{i}][{j}]", sub_sub_var_type])
                        else:
                            if sub_var_type not in var_dict.keys():
                                var_dict[sub_var_type] = {}
                            sub_ndim = len(sub_var_shape)
                            if sub_ndim not in var_dict[sub_var_type]:
                                var_dict[sub_var_type][sub_ndim] = {}
                            if str(sub_var_shape) not in var_dict[sub_var_type][sub_ndim]:
                                var_dict[sub_var_type][sub_ndim][str(sub_var_shape)] = []
                            # import pdb;pdb.set_trace()
                            var_dict[sub_var_type][sub_ndim][str(sub_var_shape)].append([var_name + f"[{i}]", var_node[i].struct_info.dtype])
                            all_pure_var_name.append([var_name + f"[{i}]", sub_var_type])

            all_var_constraints.append(var_dict)
    except Exception as e:
        print(e)
    return all_var_constraints, all_pure_var_name


if __name__ == '__main__':
    base_irs = '''# from tvm.script import ir as I
# from tvm.script import relax as R

# @I.ir_module
# class Module:
#     @T.prim_func(private=True)
#     def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
#         T.func_attr({"op_pattern": 0})
#         T.evaluate(0)
# 
#     @R.function
#     def foo(x: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(R.Tensor((2, 3), dtype="float32"),
#                                                               R.Tensor((2, 3), dtype="float32")):
#         cls = Module
#         a = R.call_tir(cls.exp, (x,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
#         with R.dataflow():
# 
#             b = R.call_tir(cls.exp, (a,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
#             c = R.call_dps_packed("packed_dps", (a,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
#             R.output(b, c)
#         return (b, c)

# @I.ir_module
# class Module:
#     @R.function
#     def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")) -> R.Tuple(R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((2, 4, 26, 26), dtype="float32")):
#         with R.dataflow():
#             gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(x, w, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
#             gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.relu(gv)
#             gv3: R.Tensor((2, 8, 26, 26), dtype="float32") = R.concat((gv, gv2), axis=1)
#             gv4: R.Tuple(R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((2, 4, 26, 26), dtype="float32")) = R.split(gv3, indices_or_sections=2, axis=1)
#             gv5 = gv4[0]
#             R.output(gv4)
#         return gv4


# @I.ir_module
# class Module:
#     @R.function
#     def prim_values() -> R.Prim(value=3):
#         x: R.Prim(value=1) = R.prim_value(1)
#         y: R.Prim(value=2) = R.prim_value(2)
#         z: R.Prim(value=3) = R.prim_value(3)
#         return z
# 
#     @R.function
#     def shapes() -> R.Shape([7, 8, 9]):
#         s1: R.Shape([1, 2, 3]) = R.shape([1, 2, 3])
#         s2: R.Shape([4, 5, 6]) = R.shape([4, 5, 6])
#         s3: R.Shape([7, 8, 9]) = R.shape([7, 8, 9])
#         return s3
# 
#     @R.function
#     def tuples_and_const(x: R.Tensor, y: R.Tensor) -> R.Tensor((3,), dtype="int32"):
#         t1: R.Tuple(R.Tensor, R.Tensor, R.Tensor) = x, y, x
#         t2: R.Tuple(R.Tensor, R.Tensor, R.Tensor) = y, y, x
#         return t2


# @I.ir_module
# class Module:
#     @T.prim_func
#     def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
#         T.evaluate(0)
# 
#     @R.function
#     def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
#         R.func_attr({"relax.force_pure": 1})
#         cls = Module
#         alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
#         cls.exp(x, alloc)
#         y1: R.Tensor((2, 3), dtype="float32") = alloc
#         alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
#         cls.exp(x, alloc1)
#         y2: R.Tensor((2, 3), dtype="float32") = alloc1
#         alloc2: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
#         cls.exp(x, alloc2)
#         y3: R.Tensor((2, 3), dtype="float32") = alloc2
#         t: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = y1, y2
#         nt: R.Tuple(R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")), R.Tensor((2, 3), dtype="float32")) = t, y3
#         nt0: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = nt[0]
#         y1_: R.Tensor((2, 3), dtype="float32") = nt0[0]
#         y2_: R.Tensor((2, 3), dtype="float32") = nt0[1]
#         y3_: R.Tensor((2, 3), dtype="float32") = nt[1]
#         alloc3: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
#         cls.exp(y1_, alloc3)
#         z1: R.Tensor((2, 3), dtype="float32") = alloc3
#         alloc4: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
#         cls.exp(y2_, alloc4)
#         z2: R.Tensor((2, 3), dtype="float32") = alloc4
#         alloc5: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"), R.prim_value(0), R.str("global"))
#         cls.exp(y3_, alloc5)
#         z3: R.Tensor((2, 3), dtype="float32") = alloc5
#         return x



@I.ir_module
class Module:
    @R.function
    def main_1(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        # from tvm.script import relax as R

        @R.function
        def outer_func38(c1: R.Tensor((2, 3), dtype="float32")) -> R.Callable((R.Tensor((2, 3), dtype="float32"),), R.Tensor((2, 3), dtype="float32"), True):
            # from tvm.script import relax as R

            @R.function
            def inner_func(x1: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
                s: R.Tensor((2, 3), dtype="float32") = R.add(x1, c1)
                return s

            return inner_func

        in_call: R.Callable((R.Tensor((2, 3), dtype="float32"),), R.Tensor((2, 3), dtype="float32"), True) = outer_func38(x)
        res: R.Tensor((2, 3), dtype="float32") = in_call(y)
        return res

    @R.function
    def main(x: R.Tensor((3,), dtype="int64")) -> R.Tensor((3,), dtype="int64"):
        lv: R.Shape([3]) = R.tensor_to_shape(x)
        gv: R.Tensor((3,), dtype="int64") = R.reshape(x, lv)
        pad_tensor = R.zeros(R.shape([3]), dtype='float32')
        gv = R.astype(gv, dtype='float32')
        tensor_1dim = R.reshape(gv, [3])
        pad_tensor = R.zeros(R.shape([3]), dtype='float32')
        temp = R.concat((tensor_1dim, pad_tensor), axis=-1)
        y = R.reshape(temp, [2, 3])
        return y

    '''
    base_irs = tvm.script.from_source(base_irs)
    print('*' * 1000)
    base_irs.show()
    seed_connection_func = base_irs["main"]
    res = collect_all_vars(seed_connection_func)
    print(res[0])
    # print(res[0]['list'][2])

# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def relax_relu_replacement(arg0: T.Buffer((16,), "float32"), output: T.Buffer((16,), "float32")):
        T.func_attr({"operator_name": "relax.relu"})
        # with T.block("root"):
        for ax0 in range(16):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(16, ax0)
                T.reads(arg0[v_ax0])
                T.writes(output[v_ax0])
                output[v_ax0] = T.max(arg0[v_ax0], T.float32(0))

    @T.prim_func(private=True)
    def relu(arg0: T.Buffer((14,), "float32"), output: T.Buffer((14,), "float32")):
        T.func_attr({"operator_name": "relax.relu"})
        # with T.block("root"):
        for ax0 in range(14):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(14, ax0)
                T.reads(arg0[v_ax0])
                T.writes(output[v_ax0])
                output[v_ax0] = T.max(arg0[v_ax0], T.float32(0))

    @T.prim_func(private=True)
    def remove_pad(var_input: T.handle, var_output: T.handle):
        T.func_attr({"operator_name": "remove_pad", "tir.noalias": T.bool(True)})
        p0 = T.int64()
        input = T.match_buffer(var_input, (p0,))
        i0 = T.int64()
        output = T.match_buffer(var_output, (i0,))
        # with T.block("root"):
        for ax0 in range(i0):
            with T.block("output"):
                v_ax0 = T.axis.spatial(i0, ax0)
                T.reads(input[v_ax0])
                T.writes(output[v_ax0])
                output[v_ax0] = input[v_ax0]

    @R.function
    def foo(x: R.Tensor((14,), dtype="float32")) -> R.Tensor((14,), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((16,), dtype="float32") = R.layout_transform(x, index_map=T.index_map(lambda i: (i % 16,)), pad_value=None, axis_separators=[])
            lv1 = R.call_tir(cls.relax_relu_replacement, (lv,), out_sinfo=R.Tensor((16,), dtype="float32"))
            lv2: R.Tensor((16,), dtype="float32") = R.layout_transform(lv1, index_map=T.index_map(lambda axis0: (axis0,)), pad_value=None, axis_separators=[])
            lv_1 = R.call_tir(cls.remove_pad, (lv2,), out_sinfo=R.Tensor((14,), dtype="float32"))
            gv: R.Tensor((14,), dtype="float32") = lv_1
            R.output(gv)
        return gv
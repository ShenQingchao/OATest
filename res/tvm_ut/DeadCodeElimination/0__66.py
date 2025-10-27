# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def relax_some_op_replacement(arg0: T.Buffer((4, 4), "float32"), arg1: T.Buffer((4, 4), "float32"), output0: T.Buffer((4, 4), "float32"), output1: T.Buffer((4, 4), "float32")):
        T.func_attr({"operator_name": "relax.some_op"})
        # with T.block("root"):
        for ax0, ax1 in T.grid(4, 4):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                T.writes(output0[v_ax0, v_ax1], output1[v_ax0, v_ax1])
                output0[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]
                output1[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] - arg1[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def some_op(arg0: T.Buffer((16,), "float32"), arg1: T.Buffer((16,), "float32"), output0: T.Buffer((16,), "float32"), output1: T.Buffer((16,), "float32")):
        T.func_attr({"operator_name": "relax.some_op"})
        # with T.block("root"):
        for ax0 in range(16):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(16, ax0)
                T.reads(arg0[v_ax0], arg1[v_ax0])
                T.writes(output0[v_ax0], output1[v_ax0])
                output0[v_ax0] = arg0[v_ax0] + arg1[v_ax0]
                output1[v_ax0] = arg0[v_ax0] - arg1[v_ax0]

    @R.function
    def main(x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tuple(R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(x, index_map=T.index_map(lambda i: (i // 4, i % 4)), pad_value=None, axis_separators=[])
            lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(y, index_map=T.index_map(lambda i: (i // 4, i % 4)), pad_value=None, axis_separators=[])
            lv2 = R.call_tir(cls.relax_some_op_replacement, (lv, lv1), out_sinfo=[R.Tensor((4, 4), dtype="float32"), R.Tensor((4, 4), dtype="float32")])
            lv3: R.Tensor((4, 4), dtype="float32") = lv2[0]
            lv4: R.Tensor((16,), dtype="float32") = R.layout_transform(lv3, index_map=T.index_map(lambda axis0, axis1: (axis0 * 4 + axis1,)), pad_value=None, axis_separators=[])
            lv5: R.Tensor((4, 4), dtype="float32") = lv2[1]
            lv6: R.Tensor((16,), dtype="float32") = R.layout_transform(lv5, index_map=T.index_map(lambda axis0, axis1: (axis0 * 4 + axis1,)), pad_value=None, axis_separators=[])
            gv: R.Tuple(R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")) = lv4, lv6
            R.output(gv)
        return gv
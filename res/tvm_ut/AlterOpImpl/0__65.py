# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
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
            gv = R.call_tir(cls.some_op, (x, y), out_sinfo=[R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")])
            R.output(gv)
        return gv
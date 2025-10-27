# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def mul_by_2(arg0: T.Buffer((16,), "float32"), output: T.Buffer((16,), "float32")):
        T.func_attr({"operator_name": "relax.mul_by_2"})
        # with T.block("root"):
        for ax0 in range(16):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(16, ax0)
                T.reads(arg0[v_ax0])
                T.writes(output[v_ax0])
                output[v_ax0] = arg0[v_ax0] * T.float32(2)

    @R.function
    def main(x: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.mul_by_2, (x,), out_sinfo=R.Tensor((16,), dtype="float32"))
            gv: R.Tensor((16,), dtype="float32") = lv
            R.output(gv)
        return gv
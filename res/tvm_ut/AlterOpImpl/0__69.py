# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
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

    @R.function
    def foo(x: R.Tensor((14,), dtype="float32")) -> R.Tensor((14,), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.relu, (x,), out_sinfo=R.Tensor((14,), dtype="float32"))
            gv: R.Tensor((14,), dtype="float32") = lv
            R.output(gv)
        return gv
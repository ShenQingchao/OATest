# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(x_0: T.Buffer((T.int64(2),), "float32"), param_0: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(2),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(2)):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(T.int64(2), ax0)
                T.reads(x_0[v_ax0], param_0[()])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = x_0[v_ax0] + param_0[()]

    @T.prim_func(private=True)
    def divide(y0: T.Buffer((T.int64(2),), "float32"), param_1: T.Buffer((), "float32"), T_divide: T.Buffer((T.int64(2),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(2)):
            with T.block("T_divide"):
                v_ax0 = T.axis.spatial(T.int64(2), ax0)
                T.reads(y0[v_ax0], param_1[()])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = y0[v_ax0] / param_1[()]

    @R.function(private=True)
    def fused_add_divide(x_0: R.Tensor((2,), dtype="float32"), param_0: R.Tensor((), dtype="float32"), param_1: R.Tensor((), dtype="float32")) -> R.Tensor((2,), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            y0 = R.call_tir(cls.add, (x_0, param_0), out_sinfo=R.Tensor((2,), dtype="float32"))
            gv = R.call_tir(cls.divide, (y0, param_1), out_sinfo=R.Tensor((2,), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tuple(R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"))) -> R.Tensor((2,), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((2,), dtype="float32") = x[0]
            lv1: R.Tensor((2,), dtype="float32") = cls.fused_add_divide(lv, R.const(1, "float32"), R.const(1, "float32"))
            gv: R.Tensor((2,), dtype="float32") = lv1
            R.output(gv)
        return gv
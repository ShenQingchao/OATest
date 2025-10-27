# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(x0: T.Buffer((T.int64(2),), "float32"), B: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(2),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(2)):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(T.int64(2), ax0)
                T.reads(x0[v_ax0], B[()])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = x0[v_ax0] + B[()]

    @T.prim_func(private=True)
    def divide(y0: T.Buffer((T.int64(2),), "float32"), B: T.Buffer((), "float32"), T_divide: T.Buffer((T.int64(2),), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(2)):
            with T.block("T_divide"):
                v_ax0 = T.axis.spatial(T.int64(2), ax0)
                T.reads(y0[v_ax0], B[()])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = y0[v_ax0] / B[()]

    @R.function
    def main(x: R.Tuple(R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"), R.Tensor((2,), dtype="float32"))) -> R.Tensor((2,), dtype="float32"):
        cls = Module
        with R.dataflow():
            x0: R.Tensor((2,), dtype="float32") = x[0]
            y0 = R.call_tir(cls.add, (x0, R.const(1, "float32")), out_sinfo=R.Tensor((2,), dtype="float32"))
            y1 = R.call_tir(cls.divide, (y0, R.const(1, "float32")), out_sinfo=R.Tensor((2,), dtype="float32"))
            gv: R.Tensor((2,), dtype="float32") = y1
            R.output(gv)
        return gv
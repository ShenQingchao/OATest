# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(x: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float32"), x_1: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], x_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] + x_1[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func(private=True)
    def foo(X: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float32"), Y: T.Buffer((T.int64(2048), T.int64(128)), "float32"), rotary: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float32"), m: T.int64):
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
            with T.block("rotary"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(Y[m + v1 - T.int64(1), v3], X[v0, v1, v2, v3])
                T.writes(rotary[v0, v1, v2, v3])
                rotary[v0, v1, v2, v3] = Y[m + v1 - T.int64(1), v3] * X[v0, v1, v2, v3]

    @R.function
    def fused(x: R.Tensor((1, 1, 32, 128), dtype="float32"), y: R.Tensor((2048, 128), dtype="float32"), len: R.Shape(["m"])) -> R.Tensor((1, 1, 32, 128), dtype="float32"):
        m = T.int64()
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv1 = R.call_tir(cls.add, (x, x), out_sinfo=R.Tensor((1, 1, 32, 128), dtype="float32"))
            gv = R.call_tir(cls.foo, (lv1, y), out_sinfo=R.Tensor((1, 1, 32, 128), dtype="float32"), tir_vars=R.shape([m]))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((1, 1, 32, 128), dtype="float32"), y: R.Tensor((2048, 128), dtype="float32"), len: R.Shape(["m"])) -> R.Tensor((1, 1, 32, 128), dtype="float32"):
        m = T.int64()
        cls = Module
        with R.dataflow():
            gv: R.Tensor((1, 1, 32, 128), dtype="float32") = cls.fused(x, y, len)
            R.output(gv)
        return gv
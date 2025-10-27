# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(var_x: T.handle, var_x_1: T.handle, var_T_add: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        sequence_length = T.int64()
        x = T.match_buffer(var_x, (T.int64(1), sequence_length, T.int64(32), T.int64(128)))
        x_1 = T.match_buffer(var_x_1, (T.int64(1), sequence_length, T.int64(32), T.int64(128)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), sequence_length, T.int64(32), T.int64(128)))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), sequence_length, T.int64(32), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], x_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] + x_1[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func(private=True)
    def foo(X_handle: T.handle, Y: T.Buffer((T.int64(2048), T.int64(128)), "float32"), rotary_handle: T.handle, m: T.int64):
        sequence_length = T.int64()
        X = T.match_buffer(X_handle, (T.int64(1), sequence_length, T.int64(32), T.int64(128)))
        rotary = T.match_buffer(rotary_handle, (T.int64(1), sequence_length, T.int64(32), T.int64(128)))
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), sequence_length, T.int64(32), T.int64(128)):
            with T.block("rotary"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(Y[m + v1 - T.int64(1), v3], X[v0, v1, v2, v3])
                T.writes(rotary[v0, v1, v2, v3])
                rotary[v0, v1, v2, v3] = Y[m + v1 - T.int64(1), v3] * X[v0, v1, v2, v3]

    @R.function
    def fused(x: R.Tensor((1, "sequence_length", 32, 128), dtype="float32"), y: R.Tensor((2048, 128), dtype="float32"), len: R.Shape(["m"])) -> R.Tensor((1, "sequence_length", 32, 128), dtype="float32"):
        sequence_length = T.int64()
        m = T.int64()
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv1 = R.call_tir(cls.add, (x, x), out_sinfo=R.Tensor((1, sequence_length, 32, 128), dtype="float32"))
            gv = R.call_tir(cls.foo, (lv1, y), out_sinfo=R.Tensor((1, sequence_length, 32, 128), dtype="float32"), tir_vars=R.shape([m]))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((1, "sequence_length", 32, 128), dtype="float32"), y: R.Tensor((2048, 128), dtype="float32"), len: R.Shape(["m"])) -> R.Tensor((1, "sequence_length", 32, 128), dtype="float32"):
        sequence_length = T.int64()
        m = T.int64()
        cls = Module
        with R.dataflow():
            gv: R.Tensor((1, sequence_length, 32, 128), dtype="float32") = cls.fused(x, y, len)
            R.output(gv)
        return gv
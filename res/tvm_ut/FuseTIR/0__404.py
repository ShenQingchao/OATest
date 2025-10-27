# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def matmul(var_y: T.handle, lv: T.Buffer((T.int64(4), T.int64(3)), "float32"), var_T_matmul: T.handle, n: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
        y = T.match_buffer(var_y, (n - T.int64(1), T.int64(4)))
        T_matmul = T.match_buffer(var_T_matmul, (n - T.int64(1), T.int64(3)))
        # with T.block("root"):
        for ax0, ax1, k in T.grid(n - T.int64(1), T.int64(3), T.int64(4)):
            with T.block("T_matmul"):
                v_ax0, v_ax1, v_k = T.axis.remap("SSR", [ax0, ax1, k])
                T.reads(y[v_ax0, v_k], lv[v_k, v_ax1])
                T.writes(T_matmul[v_ax0, v_ax1])
                with T.init():
                    T_matmul[v_ax0, v_ax1] = T.float32(0)
                T_matmul[v_ax0, v_ax1] = T_matmul[v_ax0, v_ax1] + y[v_ax0, v_k] * lv[v_k, v_ax1]

    @T.prim_func(private=True)
    def transpose(x: T.Buffer((T.int64(3), T.int64(4)), "float32"), T_transpose: T.Buffer((T.int64(4), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(4), T.int64(3)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = x[v_ax1, v_ax0]

    @R.function
    def fused_transpose_matmul(x: R.Tensor((3, 4), dtype="float32"), y: R.Tensor(("n - 1", 4), dtype="float32"), tir_vars: R.Shape(["n"])) -> R.Tensor(("n - 1", 3), dtype="float32"):
        n = T.int64()
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.transpose, (x,), out_sinfo=R.Tensor((4, 3), dtype="float32"))
            gv = R.call_tir(cls.matmul, (y, lv), out_sinfo=R.Tensor((n - 1, 3), dtype="float32"), tir_vars=R.shape([n]))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((3, 4), dtype="float32"), y: R.Tensor(("n - 1", 4), dtype="float32"), tir_vars: R.Shape(["n"])) -> R.Tensor(("n - 1", 3), dtype="float32"):
        n = T.int64()
        cls = Module
        with R.dataflow():
            lv: R.Tensor((n - 1, 3), dtype="float32") = cls.fused_transpose_matmul(x, y, tir_vars)
            R.output(lv)
        return lv
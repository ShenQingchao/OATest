# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(gv: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32"), Bias: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32"), T_add: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(32), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(gv[v_ax0, v_ax1, v_ax2], Bias[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = gv[v_ax0, v_ax1, v_ax2] + Bias[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def matmul(A: T.Buffer((T.int64(16), T.int64(32), T.int64(64)), "float32"), Weight: T.Buffer((T.int64(64), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(16), T.int64(32), T.int64(128), T.int64(64)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], Weight[v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * Weight[v_k, v_i2]

    @R.function
    def main(A: R.Tensor((16, 32, 64), dtype="float32"), Weight: R.Tensor((64, 128), dtype="float32"), Bias: R.Tensor((16, 32, 128), dtype="float32")) -> R.Tensor((16, 32, 128), dtype="float32"):
        cls = Module
        gv = R.call_tir(cls.matmul, (A, Weight), out_sinfo=R.Tensor((16, 32, 128), dtype="float32"))
        gv_1 = R.call_tir(cls.add, (gv, Bias), out_sinfo=R.Tensor((16, 32, 128), dtype="float32"))
        return gv_1
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv: T.Buffer((T.int64(1), T.int64(32), T.int64(34560)), "float32"), compute: T.Buffer((T.int64(1), T.int64(32), T.int64(1)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        rxplaceholder_red = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        for ax0, ax1, ax2, k2 in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(34560)):
            with T.block("rxplaceholder_red"):
                v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                T.reads(lv[v_ax0, v_ax1, v_k2])
                T.writes(rxplaceholder_red[v_ax0, v_ax1, v_ax2])
                with T.init():
                    rxplaceholder_red[v_ax0, v_ax1, v_ax2] = T.float32(0)
                rxplaceholder_red[v_ax0, v_ax1, v_ax2] = rxplaceholder_red[v_ax0, v_ax1, v_ax2] + lv[v_ax0, v_ax1, v_k2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(1)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(rxplaceholder_red[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = rxplaceholder_red[v_ax0, v_ax1, v_ax2] * T.float32(2.8935185185185186e-05)
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(32), T.int64(1)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_divide[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float16", T_divide[v_i0, v_i1, v_i2])
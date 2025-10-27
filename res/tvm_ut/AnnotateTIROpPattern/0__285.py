# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(x: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16"), B: T.Buffer((), "float16"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], B[()])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] + B[()]

    @T.prim_func(private=True)
    def add1(A: T.Buffer((), "float16"), lv1: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[()], lv1[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[()] + lv1[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func(private=True)
    def add2(lv1: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16"), lv2: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv1[v_ax0, v_ax1, v_ax2, v_ax3], lv2[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = lv1[v_ax0, v_ax1, v_ax2, v_ax3] + lv2[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func(private=True)
    def conv2d(lv: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16"), w1: T.Buffer((T.int64(16), T.int64(16), T.int64(3), T.int64(3)), "float16"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(66), T.int64(66)), "float16")
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(66), T.int64(66)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), lv[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float16(0))
        for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64), T.int64(16), T.int64(3), T.int64(3)):
            with T.block("conv2d_nchw"):
                v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], w1[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float16(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * w1[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d1(lv3: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16"), w2: T.Buffer((T.int64(16), T.int64(16), T.int64(1), T.int64(1)), "float16"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float16")
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv3[v_i0, v_i1, v_i2, v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = lv3[v_i0, v_i1, v_i2, v_i3]
        for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64), T.int64(16), T.int64(1), T.int64(1)):
            with T.block("conv2d_nchw"):
                v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], w2[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float16(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * w2[v_ff, v_rc, v_ry, v_rx]

    @R.function
    def main(x: R.Tensor((1, 16, 64, 64), dtype="float16"), w1: R.Tensor((16, 16, 3, 3), dtype="float16"), w2: R.Tensor((16, 16, 1, 1), dtype="float16"), w3: R.Tensor((16, 16, 3, 3), dtype="float16")) -> R.Tensor((1, 16, 64, 64), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.add, (x, R.const(1, "float16")), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float16"))
            lv1 = R.call_tir(cls.conv2d, (lv, w1), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float16"))
            lv2 = R.call_tir(cls.add1, (R.const(1, "float16"), lv1), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float16"))
            lv3 = R.call_tir(cls.add2, (lv1, lv2), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float16"))
            lv4 = R.call_tir(cls.conv2d1, (lv3, w2), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float16"))
            lv5 = R.call_tir(cls.conv2d, (lv3, w3), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float16"))
            gv = R.call_tir(cls.add2, (lv4, lv5), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float16"))
            R.output(gv)
        return gv
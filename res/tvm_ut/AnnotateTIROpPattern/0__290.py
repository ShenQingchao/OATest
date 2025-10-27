# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(x: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8"), B: T.Buffer((), "int8"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], B[()])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] + B[()]

    @T.prim_func(private=True)
    def add1(p0: T.Buffer((), "int8"), lv: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(p0[()], lv[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = p0[()] + lv[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func(private=True)
    def add2(lv: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8"), lv1: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv[v_ax0, v_ax1, v_ax2, v_ax3], lv1[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = lv[v_ax0, v_ax1, v_ax2, v_ax3] + lv1[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func(private=True)
    def conv2d(x: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8"), w: T.Buffer((T.int64(16), T.int64(16), T.int64(3), T.int64(3)), "int8"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(66), T.int64(66)), "int8")
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(66), T.int64(66)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(x[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), x[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.int8(0))
        for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64), T.int64(16), T.int64(3), T.int64(3)):
            with T.block("conv2d_nchw"):
                v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], w[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.int8(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * w[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d1(x: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8"), w: T.Buffer((T.int64(16), T.int64(16), T.int64(1), T.int64(1)), "int8"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "int8")
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(x[v_i0, v_i1, v_i2, v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = x[v_i0, v_i1, v_i2, v_i3]
        for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64), T.int64(16), T.int64(1), T.int64(1)):
            with T.block("conv2d_nchw"):
                v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], w[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.int8(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * w[v_ff, v_rc, v_ry, v_rx]

    @R.function(private=True)
    def fused_conv2d1_add2(x: R.Tensor((1, 16, 64, 64), dtype="int8"), w: R.Tensor((16, 16, 1, 1), dtype="int8"), y: R.Tensor((1, 16, 64, 64), dtype="int8")) -> R.Tensor((1, 16, 64, 64), dtype="int8"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv2 = R.call_tir(cls.conv2d1, (x, w), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="int8"))
            gv1 = R.call_tir(cls.add2, (lv2, y), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="int8"))
            R.output(gv1)
        return gv1

    @R.function(private=True)
    def fused_conv2d_add1_add2(x: R.Tensor((1, 16, 64, 64), dtype="int8"), w: R.Tensor((16, 16, 3, 3), dtype="int8"), p0: R.Tensor((), dtype="int8")) -> R.Tensor((1, 16, 64, 64), dtype="int8"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.conv2d, (x, w), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="int8"))
            lv1 = R.call_tir(cls.add1, (p0, lv), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="int8"))
            gv = R.call_tir(cls.add2, (lv, lv1), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="int8"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((1, 16, 64, 64), dtype="int8"), w1: R.Tensor((16, 16, 3, 3), dtype="int8"), w2: R.Tensor((16, 16, 1, 1), dtype="int8"), w3: R.Tensor((16, 16, 3, 3), dtype="int8")) -> R.Tensor((1, 16, 64, 64), dtype="int8"):
        cls = Module
        with R.dataflow():
            lv3 = R.call_tir(cls.add, (x, R.const(1, "int8")), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="int8"))
            lv4: R.Tensor((1, 16, 64, 64), dtype="int8") = cls.fused_conv2d_add1_add2(lv3, w1, R.const(1, "int8"))
            lv5 = R.call_tir(cls.conv2d, (lv4, w3), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="int8"))
            gv2: R.Tensor((1, 16, 64, 64), dtype="int8") = cls.fused_conv2d1_add2(lv4, w2, lv5)
            R.output(gv2)
        return gv2
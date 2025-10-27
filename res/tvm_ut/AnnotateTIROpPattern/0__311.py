# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def concatenate(lv2: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32"), lv3: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32"), T_concat: T.Buffer((T.int64(1), T.int64(32), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(64), T.int64(64)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv3[v_ax0, v_ax1 - T.int64(16), v_ax2, v_ax3], lv2[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
                T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(16) <= v_ax1, lv3[v_ax0, v_ax1 - T.int64(16), v_ax2, v_ax3], lv2[v_ax0, v_ax1, v_ax2, v_ax3])

    @T.prim_func(private=True)
    def conv2d(x: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32"), w: T.Buffer((T.int64(16), T.int64(16), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(66), T.int64(66)))
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(66), T.int64(66)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(x[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), x[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64), T.int64(16), T.int64(3), T.int64(3)):
            with T.block("conv2d_nchw"):
                v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], w[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * w[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d1(x: T.Buffer((T.int64(1), T.int64(32), T.int64(64), T.int64(64)), "float32"), w: T.Buffer((T.int64(16), T.int64(32), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(66), T.int64(66)))
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(66), T.int64(66)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(x[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), x[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64), T.int64(32), T.int64(3), T.int64(3)):
            with T.block("conv2d_nchw"):
                v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], w[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * w[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def relu(lv: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32"), compute: T.Buffer((T.int64(1), T.int64(16), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(64), T.int64(64)):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.max(lv[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    @R.function(private=True)
    def fused_conv2d1_relu(x: R.Tensor((1, 32, 64, 64), dtype="float32"), w: R.Tensor((16, 32, 3, 3), dtype="float32")) -> R.Tensor((1, 16, 64, 64), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv1 = R.call_tir(cls.conv2d1, (x, w), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            gv1 = R.call_tir(cls.relu, (lv1,), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            R.output(gv1)
        return gv1

    @R.function(private=True)
    def fused_conv2d_relu(x: R.Tensor((1, 16, 64, 64), dtype="float32"), w: R.Tensor((16, 16, 3, 3), dtype="float32")) -> R.Tensor((1, 16, 64, 64), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.conv2d, (x, w), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            gv = R.call_tir(cls.relu, (lv,), out_sinfo=R.Tensor((1, 16, 64, 64), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((1, 16, 64, 64), dtype="float32"), w0: R.Tensor((16, 16, 3, 3), dtype="float32"), w1: R.Tensor((16, 16, 3, 3), dtype="float32"), w2: R.Tensor((16, 32, 3, 3), dtype="float32"), w3: R.Tensor((16, 32, 3, 3), dtype="float32")) -> R.Tensor((1, 32, 64, 64), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv2: R.Tensor((1, 16, 64, 64), dtype="float32") = cls.fused_conv2d_relu(x, w0)
            lv3: R.Tensor((1, 16, 64, 64), dtype="float32") = cls.fused_conv2d_relu(x, w1)
            lv4 = R.call_tir(cls.concatenate, (lv2, lv3), out_sinfo=R.Tensor((1, 32, 64, 64), dtype="float32"))
            lv5: R.Tensor((1, 16, 64, 64), dtype="float32") = cls.fused_conv2d1_relu(lv4, w2)
            lv6: R.Tensor((1, 16, 64, 64), dtype="float32") = cls.fused_conv2d1_relu(lv4, w3)
            gv2 = R.call_tir(cls.concatenate, (lv5, lv6), out_sinfo=R.Tensor((1, 32, 64, 64), dtype="float32"))
            R.output(gv2)
        return gv2
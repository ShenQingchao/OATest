# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv27: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32"), lv28: T.Buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv27[v_ax0, v_ax1, v_ax2, v_ax3], lv28[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = lv27[v_ax0, v_ax1, v_ax2, v_ax3] + lv28[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func(private=True)
    def add1(lv32: T.Buffer((T.int64(2), T.int64(320)), "float32"), b2: T.Buffer((T.int64(320),), "float32"), T_add: T.Buffer((T.int64(2), T.int64(320)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(2), T.int64(320)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv32[v_ax0, v_ax1], b2[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv32[v_ax0, v_ax1] + b2[v_ax1]

    @T.prim_func(private=True)
    def add2(lv29: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32"), lv35: T.Buffer((T.int64(2), T.int64(320), T.int64(1), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv29[v_ax0, v_ax1, v_ax2, v_ax3], lv35[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = lv29[v_ax0, v_ax1, v_ax2, v_ax3] + lv35[v_ax0, v_ax1, T.int64(0), T.int64(0)]

    @T.prim_func(private=True)
    def conv2d(inp_0: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32"), w1: T.Buffer((T.int64(320), T.int64(320), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(66), T.int64(66)))
        for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(320), T.int64(66), T.int64(66)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inp_0[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), inp_0[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
        for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64), T.int64(320), T.int64(3), T.int64(3)):
            with T.block("conv2d_nchw"):
                v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], w1[v_ff, v_rc, v_ry, v_rx])
                T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                with T.init():
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * w1[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def matmul(inp_1: T.Buffer((T.int64(2), T.int64(1280)), "float32"), lv31: T.Buffer((T.int64(1280), T.int64(320)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(320)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(2), T.int64(320), T.int64(1280)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(inp_1[v_i0, v_k], lv31[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + inp_1[v_i0, v_k] * lv31[v_k, v_i1]

    @T.prim_func(private=True)
    def reshape(b1: T.Buffer((T.int64(320),), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(320), T.int64(1), T.int64(1)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(b1[(v_ax1 + v_ax2 + v_ax3) % T.int64(320)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = b1[(v_ax1 + v_ax2 + v_ax3) % T.int64(320)]

    @T.prim_func(private=True)
    def reshape1(lv33: T.Buffer((T.int64(2), T.int64(320)), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(320), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(1), T.int64(1)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv33[((v_ax1 + v_ax2 + v_ax3) // T.int64(320) + v_ax0) % T.int64(2), (v_ax1 + v_ax2 + v_ax3) % T.int64(320)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = lv33[((v_ax1 + v_ax2 + v_ax3) // T.int64(320) + v_ax0) % T.int64(2), (v_ax1 + v_ax2 + v_ax3) % T.int64(320)]

    @T.prim_func(private=True)
    def transpose(w2: T.Buffer((T.int64(320), T.int64(1280)), "float32"), T_transpose: T.Buffer((T.int64(1280), T.int64(320)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1280), T.int64(320)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(w2[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = w2[v_ax1, v_ax0]

    @R.function
    def main(inp_0: R.Tensor((2, 320, 64, 64), dtype="float32"), inp_1: R.Tensor((2, 1280), dtype="float32"), w1: R.Tensor((320, 320, 3, 3), dtype="float32"), b1: R.Tensor((320,), dtype="float32"), w2: R.Tensor((320, 1280), dtype="float32"), b2: R.Tensor((320,), dtype="float32")) -> R.Tensor((2, 320, 64, 64), dtype="float32"):
        R.func_attr({"num_input": 2})
        cls = Module
        with R.dataflow():
            lv27 = R.call_tir(cls.conv2d, (inp_0, w1), out_sinfo=R.Tensor((2, 320, 64, 64), dtype="float32"))
            lv28 = R.call_tir(cls.reshape, (b1,), out_sinfo=R.Tensor((1, 320, 1, 1), dtype="float32"))
            lv29 = R.call_tir(cls.add, (lv27, lv28), out_sinfo=R.Tensor((2, 320, 64, 64), dtype="float32"))
            lv31 = R.call_tir(cls.transpose, (w2,), out_sinfo=R.Tensor((1280, 320), dtype="float32"))
            lv32 = R.call_tir(cls.matmul, (inp_1, lv31), out_sinfo=R.Tensor((2, 320), dtype="float32"))
            lv33 = R.call_tir(cls.add1, (lv32, b2), out_sinfo=R.Tensor((2, 320), dtype="float32"))
            lv35 = R.call_tir(cls.reshape1, (lv33,), out_sinfo=R.Tensor((2, 320, 1, 1), dtype="float32"))
            lv36 = R.call_tir(cls.add2, (lv29, lv35), out_sinfo=R.Tensor((2, 320, 64, 64), dtype="float32"))
            gv: R.Tensor((2, 320, 64, 64), dtype="float32") = lv36
            R.output(gv)
        return gv
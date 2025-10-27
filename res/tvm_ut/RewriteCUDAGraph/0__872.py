# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_conv2d_relu(data: T.Buffer((T.int64(16), T.int64(32), T.int64(32), T.int64(16)), "float16"), weight1: T.Buffer((T.int64(16), T.int64(3), T.int64(3), T.int64(16)), "float16"), compute_intermediate: T.Buffer((T.int64(16), T.int64(32), T.int64(32), T.int64(16)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(16), T.int64(34), T.int64(34), T.int64(16)), "float16")
        conv2d_nhwc_intermediate = T.alloc_buffer((T.int64(16), T.int64(32), T.int64(32), T.int64(16)), "float16")
        for i0, i1, i2, i3 in T.grid(T.int64(16), T.int64(34), T.int64(34), T.int64(16)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(data[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3])
                T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i1 and v_i1 < T.int64(33) and T.int64(1) <= v_i2 and v_i2 < T.int64(33), data[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3], T.float16(0))
        for nn, yy, xx, ff, ry, rx, rc in T.grid(T.int64(16), T.int64(32), T.int64(32), T.int64(16), T.int64(3), T.int64(3), T.int64(16)):
            with T.block("conv2d_nhwc"):
                v_nn, v_yy, v_xx, v_ff, v_ry, v_rx, v_rc = T.axis.remap("SSSSRRR", [nn, yy, xx, ff, ry, rx, rc])
                T.reads(pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc], weight1[v_ff, v_ry, v_rx, v_rc])
                T.writes(conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff])
                with T.init():
                    conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff] = T.float16(0)
                conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff] = conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff] + pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc] * weight1[v_ff, v_ry, v_rx, v_rc]
        for i0, i1, i2, i3 in T.grid(T.int64(16), T.int64(32), T.int64(32), T.int64(16)):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_nhwc_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(conv2d_nhwc_intermediate[v_i0, v_i1, v_i2, v_i3], T.float16(0))

    @T.prim_func(private=True)
    def layer_norm(conv2: T.Buffer((T.int64(16), T.int64(32), T.int64(32), T.int64(16)), "float16"), gamma: T.Buffer((T.int64(16),), "float16"), beta: T.Buffer((T.int64(16),), "float16"), T_layer_norm: T.Buffer((T.int64(16), T.int64(32), T.int64(32), T.int64(16)), "float16")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        conv2_red_temp_v0 = T.alloc_buffer((T.int64(16), T.int64(32), T.int64(32)))
        conv2_red_temp_v1 = T.alloc_buffer((T.int64(16), T.int64(32), T.int64(32)))
        for ax0, ax1, ax2, k3 in T.grid(T.int64(16), T.int64(32), T.int64(32), T.int64(16)):
            with T.block("conv2_red_temp"):
                v_ax0, v_ax1, v_ax2, v_k3 = T.axis.remap("SSSR", [ax0, ax1, ax2, k3])
                T.reads(conv2[v_ax0, v_ax1, v_ax2, v_k3])
                T.writes(conv2_red_temp_v0[v_ax0, v_ax1, v_ax2], conv2_red_temp_v1[v_ax0, v_ax1, v_ax2])
                with T.init():
                    conv2_red_temp_v0[v_ax0, v_ax1, v_ax2] = T.float32(0)
                    conv2_red_temp_v1[v_ax0, v_ax1, v_ax2] = T.float32(0)
                v_conv2_red_temp_v0: T.float32 = conv2_red_temp_v0[v_ax0, v_ax1, v_ax2] + T.Cast("float32", conv2[v_ax0, v_ax1, v_ax2, v_k3])
                v_conv2_red_temp_v1: T.float32 = conv2_red_temp_v1[v_ax0, v_ax1, v_ax2] + T.Cast("float32", conv2[v_ax0, v_ax1, v_ax2, v_k3]) * T.Cast("float32", conv2[v_ax0, v_ax1, v_ax2, v_k3])
                conv2_red_temp_v0[v_ax0, v_ax1, v_ax2] = v_conv2_red_temp_v0
                conv2_red_temp_v1[v_ax0, v_ax1, v_ax2] = v_conv2_red_temp_v1
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(32), T.int64(32), T.int64(16)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(conv2[v_ax0, v_ax1, v_ax2, v_ax3], conv2_red_temp_v0[v_ax0, v_ax1, v_ax2], conv2_red_temp_v1[v_ax0, v_ax1, v_ax2], gamma[v_ax3], beta[v_ax3])
                T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3] = T.Cast("float16", (T.Cast("float32", conv2[v_ax0, v_ax1, v_ax2, v_ax3]) - conv2_red_temp_v0[v_ax0, v_ax1, v_ax2] * T.float32(0.0625)) * T.rsqrt(conv2_red_temp_v1[v_ax0, v_ax1, v_ax2] * T.float32(0.0625) - conv2_red_temp_v0[v_ax0, v_ax1, v_ax2] * T.float32(0.0625) * (conv2_red_temp_v0[v_ax0, v_ax1, v_ax2] * T.float32(0.0625)) + T.float32(1.0000000000000001e-05))) * gamma[v_ax3] + beta[v_ax3]

    @R.function
    def main_transform_params(model_params: R.Tuple(R.Tensor((16, 3, 3, 16), dtype="float16"), R.Tensor((16, 3, 3, 16), dtype="float16"), R.Tensor((16, 3, 3, 16), dtype="float16"), R.Tensor((16,), dtype="float16"), R.Tensor((16,), dtype="float16"))) -> R.Tuple(R.Tensor((16, 3, 3, 16), dtype="float16"), R.Tensor((16, 3, 3, 16), dtype="float16"), R.Tensor((16, 3, 3, 16), dtype="float16"), R.Tensor((16,), dtype="float16"), R.Tensor((16,), dtype="float16")):
        R.func_attr({"num_input": 0, "relax.force_pure": 1})
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16") = model_params[0]
        weight2: R.Tensor((16, 3, 3, 16), dtype="float16") = model_params[1]
        weight3: R.Tensor((16, 3, 3, 16), dtype="float16") = model_params[2]
        gamma: R.Tensor((16,), dtype="float16") = model_params[3]
        beta: R.Tensor((16,), dtype="float16") = model_params[4]
        output_tuple: R.Tuple(R.Tensor((16, 3, 3, 16), dtype="float16"), R.Tensor((16, 3, 3, 16), dtype="float16"), R.Tensor((16, 3, 3, 16), dtype="float16"), R.Tensor((16,), dtype="float16"), R.Tensor((16,), dtype="float16")) = weight1, weight2, weight3, gamma, beta
        return output_tuple

    @R.function
    def main(data: R.Tensor((16, 32, 32, 16), dtype="float16"), weight1: R.Tensor((16, 3, 3, 16), dtype="float16"), weight2: R.Tensor((16, 3, 3, 16), dtype="float16"), weight3: R.Tensor((16, 3, 3, 16), dtype="float16"), gamma: R.Tensor((16,), dtype="float16"), beta: R.Tensor((16,), dtype="float16")) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        R.func_attr({"num_input": 1, "relax.force_pure": 1})
        cls = Module
        storage: R.Object = R.memory.alloc_storage(R.shape([524288]), R.prim_value(0), R.str("global"), R.dtype("float16"))
        alloc: R.Tensor((16, 32, 32, 16), dtype="float16") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([16, 32, 32, 16]), R.dtype("float16"))
        cls.fused_conv2d_relu(data, weight1, alloc)
        lv: R.Tensor((16, 32, 32, 16), dtype="float16") = alloc
        storage1: R.Object = R.memory.alloc_storage(R.shape([524288]), R.prim_value(0), R.str("global"), R.dtype("float16"))
        alloc1: R.Tensor((16, 32, 32, 16), dtype="float16") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 32, 32, 16]), R.dtype("float16"))
        cls.fused_conv2d_relu(lv, weight2, alloc1)
        lv1: R.Tensor((16, 32, 32, 16), dtype="float16") = alloc1
        alloc2: R.Tensor((16, 32, 32, 16), dtype="float16") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([16, 32, 32, 16]), R.dtype("float16"))
        cls.layer_norm(lv1, gamma, beta, alloc2)
        ln: R.Tensor((16, 32, 32, 16), dtype="float16") = alloc2
        alloc3: R.Tensor((16, 32, 32, 16), dtype="float16") = R.builtin.alloc_tensor(R.shape([16, 32, 32, 16]), R.dtype("float16"), R.prim_value(0), R.str("global"))
        cls.fused_conv2d_relu(ln, weight3, alloc3)
        gv: R.Tensor((16, 32, 32, 16), dtype="float16") = alloc3
        return gv
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def layer_norm(A: T.Buffer((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32"), gamma: T.Buffer((T.int64(64), T.int64(64)), "float32"), beta: T.Buffer((T.int64(64), T.int64(64)), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 4})
        # with T.block("root"):
        rxplaceholder_red_temp_v0 = T.alloc_buffer((T.int64(64), T.int64(64)))
        rxplaceholder_red_temp_v1 = T.alloc_buffer((T.int64(64), T.int64(64)))
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(64), T.int64(64)):
            with T.block("rxplaceholder_red_temp"):
                ax0, ax1, k2, k3 = T.axis.remap("SSRR", [i0, i1, i2, i3])
                T.reads(A[ax0, ax1, k2, k3])
                T.writes(rxplaceholder_red_temp_v0[ax0, ax1], rxplaceholder_red_temp_v1[ax0, ax1])
                with T.init():
                    rxplaceholder_red_temp_v0[ax0, ax1] = T.float32(0)
                    rxplaceholder_red_temp_v1[ax0, ax1] = T.float32(0)
                v_rxplaceholder_red_temp_v0: T.float32 = rxplaceholder_red_temp_v0[ax0, ax1] + A[ax0, ax1, k2, k3]
                v_rxplaceholder_red_temp_v1: T.float32 = rxplaceholder_red_temp_v1[ax0, ax1] + A[ax0, ax1, k2, k3] * A[ax0, ax1, k2, k3]
                rxplaceholder_red_temp_v0[ax0, ax1] = v_rxplaceholder_red_temp_v0
                rxplaceholder_red_temp_v1[ax0, ax1] = v_rxplaceholder_red_temp_v1
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(64), T.int64(64)):
            with T.block("T_layer_norm"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(A[ax0, ax1, ax2, ax3], rxplaceholder_red_temp_v0[ax0, ax1], rxplaceholder_red_temp_v1[ax0, ax1], gamma[ax2, ax3], beta[ax2, ax3])
                T.writes(T_layer_norm[ax0, ax1, ax2, ax3])
                T_layer_norm[ax0, ax1, ax2, ax3] = (A[ax0, ax1, ax2, ax3] - rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.050000000000000003)) * T.rsqrt(rxplaceholder_red_temp_v1[ax0, ax1] * T.float32(0.050000000000000003) - rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.050000000000000003) * (rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.050000000000000003)) + T.float32(1.0000000000000001e-05)) * gamma[ax2, ax3] + beta[ax2, ax3]

    @T.prim_func(private=True)
    def relu(A: T.Buffer((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32"), B: T.Buffer((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 0})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(64), T.int64(64)):
            with T.block("relu"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(A[v_i0, v_i1, v_i2, v_i3])
                T.writes(B[v_i0, v_i1, v_i2, v_i3])
                B[v_i0, v_i1, v_i2, v_i3] = T.max(A[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    @R.function(private=True)
    def fused_layer_norm_relu(x: R.Tensor((1, 512, 64, 64), dtype="float32"), mean: R.Tensor((64, 64), dtype="float32"), var: R.Tensor((64, 64), dtype="float32")) -> R.Tensor((1, 512, 64, 64), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            gv0 = R.call_tir(cls.layer_norm, (x, mean, var), out_sinfo=R.Tensor((1, 512, 64, 64)))
            gv = R.call_tir(cls.relu, (gv0,), out_sinfo=R.Tensor((1, 512, 64, 64), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((1, 512, 64, 64), dtype="float32"), mean: R.Tensor((64, 64), dtype="float32"), var: R.Tensor((64, 64), dtype="float32")) -> R.Tensor((1, 512, 64, 64), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((1, 512, 64, 64), dtype="float32") = cls.fused_layer_norm_relu(x, mean, var)
            R.output(gv)
        return gv
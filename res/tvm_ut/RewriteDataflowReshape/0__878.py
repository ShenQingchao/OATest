# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def expand_dims(rxplaceholder: T.Buffer((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float32"), expand_dims: T.Buffer((T.int64(2), T.int64(1), T.int64(4096), T.int64(1), T.int64(5), T.int64(64)), "float32")):
        # with T.block("root"):
        for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(2), T.int64(1), T.int64(4096), T.int64(1), T.int64(5), T.int64(64)):
            with T.block("expand_dims"):
                i0_1, i1_1, i2_1, i3_1, i4_1, i5_1 = T.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                T.reads(rxplaceholder[i0_1, i2_1, i4_1, i5_1])
                T.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1])
                expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1] = rxplaceholder[i0_1, i2_1, i4_1, i5_1]

    @T.prim_func
    def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float32")):
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(10)):
                    with T.block("T_reshape"):
                        v_ax0 = T.axis.spatial(T.int64(2), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(1310720))
                        v_ax1 = T.axis.spatial(T.int64(4096), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(1310720) // T.int64(320))
                        v_ax2 = T.axis.spatial(T.int64(5), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(320) // T.int64(64))
                        v_ax3 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(64))
                        T.reads(rxplaceholder[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(320)])
                        T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(320)]

    @R.function
    def main(x: R.Tensor((2, 4096, 320), dtype="float32")) -> R.Tensor((2, 1, 4096, 1, 5, 64), dtype="float32"):
        cls = Module
        with R.dataflow():
            y = R.call_tir(cls.reshape, (x,), out_sinfo=R.Tensor((2, 4096, 5, 64), dtype="float32"))
            z = R.call_tir(cls.expand_dims, (y,), out_sinfo=R.Tensor((2, 1, 4096, 1, 5, 64), dtype="float32"))
            R.output(z)
        return z
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def expand_dims(rxplaceholder: T.Buffer((T.int64(2), T.int64(4), T.int64(3)), "float32"), expand_dims: T.Buffer((T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)), "float32")):
        # with T.block("root"):
        for i0, i1, i2, i3, i4 in T.grid(T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)):
            with T.block("expand_dims"):
                i0_1, i1_1, i2_1, i3_1, i4_1 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(rxplaceholder[i0_1, i2_1, i4_1])
                T.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1])
                expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1] = rxplaceholder[i0_1, i2_1, i4_1]

    @T.prim_func
    def reshape(rxplaceholder: T.Buffer((T.int64(8), T.int64(3)), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(4), T.int64(3)), "float32")):
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(3)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(rxplaceholder[(v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) // T.int64(3), (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) % T.int64(3)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[(v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) // T.int64(3), (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) % T.int64(3)]

    @R.function
    def main(x: R.Tensor((8, 3), dtype="float32")) -> R.Tensor((2, 1, 4, 1, 3), dtype="float32"):
        cls = Module
        with R.dataflow():
            y = R.call_tir(cls.reshape, (x,), out_sinfo=R.Tensor((2, 4, 3), dtype="float32"))
            z = R.call_tir(cls.expand_dims, (y,), out_sinfo=R.Tensor((2, 1, 4, 1, 3), dtype="float32"))
            R.output(z)
        return z
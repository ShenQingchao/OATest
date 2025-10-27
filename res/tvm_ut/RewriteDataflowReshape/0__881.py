# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def fused_reshape5(lv2_0: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float16"), lv2_1: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float16"), lv2_2: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float16"), T_reshape_handle_intermediate: T.Buffer((T.int64(2), T.int64(4096), T.int64(8), T.int64(40)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4096), T.int64(8), T.int64(40)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv2_0[(((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(40) + v_ax3) % T.int64(320)])
                T.writes(T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv2_0[(((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(40) + v_ax3) % T.int64(320)]

    @R.function
    def main(lv41_1: R.Tuple(R.Tensor((2, 4096, 320), dtype="float16"), R.Tensor((2, 4096, 320), dtype="float16"), R.Tensor((2, 4096, 320), dtype="float16"))) -> R.Tensor((2, 4096, 8, 40), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[0]
            lv1: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[1]
            lv2: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[2]
            lv645 = R.call_tir(cls.fused_reshape5, (lv, lv1, lv2), out_sinfo=R.Tensor((2, 4096, 8, 40), dtype="float16"))
            out: R.Tensor((2, 4096, 8, 40), dtype="float16") = R.add(lv645, lv645)
            R.output(out)
        return out
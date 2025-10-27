# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def reshape(A: T.Buffer((T.int64(4), T.int64(8), T.int64(2048)), "float32"), T_reshape: T.Buffer((T.int64(4), T.int64(8), T.int64(32), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(8), T.int64(32), T.int64(64)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1) // T.int64(8) + v_ax0) % T.int64(4), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1) % T.int64(8), (v_ax2 * T.int64(64) + v_ax3) % T.int64(2048)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1) // T.int64(8) + v_ax0) % T.int64(4), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1) % T.int64(8), (v_ax2 * T.int64(64) + v_ax3) % T.int64(2048)]

    @R.function(private=True)
    def fused_reshape(lv: R.Tuple(R.Tensor((4, 8, 2048), dtype="float32"), R.Tensor((4, 8, 2048), dtype="float32"))) -> R.Tensor((4, 8, 32, 64), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv1: R.Tensor((4, 8, 2048), dtype="float32") = lv[0]
            gv = R.call_tir(cls.reshape, (lv1,), out_sinfo=R.Tensor((4, 8, 32, 64), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(tup: R.Tuple(R.Tensor((4, 8, 2048), dtype="float32"), R.Tensor((4, 8, 2048), dtype="float32"))) -> R.Tensor((4, 8, 32, 64), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv_1: R.Tensor((4, 8, 32, 64), dtype="float32") = cls.fused_reshape(tup)
            R.output(lv_1)
        return lv_1
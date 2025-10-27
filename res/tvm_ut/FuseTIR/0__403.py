# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def concatenate(rxplaceholder: T.Buffer((T.int64(1), T.int64(4), T.int64(64), T.int64(64)), "float32"), rxplaceholder_1: T.Buffer((T.int64(1), T.int64(4), T.int64(64), T.int64(64)), "float32"), T_concat: T.Buffer((T.int64(2), T.int64(4), T.int64(64), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4), T.int64(64), T.int64(64)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(rxplaceholder_1[v_ax0 - T.int64(1), v_ax1, v_ax2, v_ax3], rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
                T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1) <= v_ax0, rxplaceholder_1[v_ax0 - T.int64(1), v_ax1, v_ax2, v_ax3], rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3])

    @T.prim_func(private=True)
    def transpose2(rxplaceholder: T.Buffer((T.int64(2), T.int64(4), T.int64(64), T.int64(64)), "float32"), T_transpose: T.Buffer((T.int64(2), T.int64(64), T.int64(64), T.int64(4)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(64), T.int64(64), T.int64(4)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(rxplaceholder[v_ax0, v_ax3, v_ax1, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax3, v_ax1, v_ax2]

    @R.function
    def fused_concatenate_transpose2(inp_0: R.Tensor((1, 4, 64, 64), dtype="float32")) -> R.Tensor((2, 64, 64, 4), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.concatenate, (inp_0, inp_0), out_sinfo=R.Tensor((2, 4, 64, 64), dtype="float32"))
            gv = R.call_tir(cls.transpose2, (lv,), out_sinfo=R.Tensor((2, 64, 64, 4), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(inp_0: R.Tensor((1, 4, 64, 64), dtype="float32")) -> R.Tensor((2, 64, 64, 4), dtype="float32"):
        R.func_attr({"num_input": 3})
        cls = Module
        with R.dataflow():
            lv: R.Tensor((2, 64, 64, 4), dtype="float32") = cls.fused_concatenate_transpose2(inp_0)
            R.output(lv)
        return lv
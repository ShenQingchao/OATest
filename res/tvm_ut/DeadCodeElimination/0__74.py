# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def relax_reshape_replacement(A: T.Buffer((T.int64(850), T.int64(2), T.int64(1024)), "float16"), T_reshape: T.Buffer((T.int64(850), T.int64(1), T.int64(2048)), "float16")):
        T.func_attr({"operator_name": "relax.reshape"})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(850), T.int64(1), T.int64(2048)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax2 // T.int64(1024), v_ax2 % T.int64(1024)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax2 // T.int64(1024), v_ax2 % T.int64(1024)]

    @T.prim_func(private=True)
    def reshape(A: T.Buffer((T.int64(850), T.int64(2048)), "float16"), T_reshape: T.Buffer((T.int64(850), T.int64(1), T.int64(2048)), "float16")):
        T.func_attr({"operator_name": "relax.reshape"})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(850), T.int64(1), T.int64(2048)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[(v_ax2 // T.int64(2048) + v_ax0 + v_ax1) % T.int64(850), v_ax2 % T.int64(2048)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax2 // T.int64(2048) + v_ax0 + v_ax1) % T.int64(850), v_ax2 % T.int64(2048)]

    @R.function
    def main(x: R.Tensor((850, 2048), dtype="float16")) -> R.Tensor((850, 1, 2048), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((850, 2, 1024), dtype="float16") = R.layout_transform(x, index_map=T.index_map(lambda i, j: (i, j // 1024, j % 1024)), pad_value=None, axis_separators=[])
            lv_1 = R.call_tir(cls.relax_reshape_replacement, (lv,), out_sinfo=R.Tensor((850, 1, 2048), dtype="float16"))
            gv: R.Tensor((850, 1, 2048), dtype="float16") = lv_1
            R.output(gv)
        return gv
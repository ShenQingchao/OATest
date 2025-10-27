# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def dense(lv1: T.Buffer((T.int64(1), T.int64(10)), "float32"), w: T.Buffer((T.int64(10), T.int64(10)), "float32"), T_matmul_NT: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"layout_free_buffers": [1], "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(10), T.int64(10)):
            with T.block("T_matmul_NT"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv1[v_i0, v_k], w[v_i1, v_k])
                T.writes(T_matmul_NT[v_i0, v_i1])
                with T.init():
                    T_matmul_NT[v_i0, v_i1] = T.float32(0)
                T_matmul_NT[v_i0, v_i1] = T_matmul_NT[v_i0, v_i1] + lv1[v_i0, v_k] * w[v_i1, v_k]

    @T.prim_func(private=True)
    def split(x: T.Buffer((T.int64(1), T.int64(30)), "float32"), T_split_sections: T.Buffer((T.int64(1), T.int64(10)), "float32"), T_split_sections_1: T.Buffer((T.int64(1), T.int64(10)), "float32"), T_split_sections_2: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_split_sections"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1])
                T.writes(T_split_sections[v_ax0, v_ax1])
                T_split_sections[v_ax0, v_ax1] = x[v_ax0, v_ax1]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_split_sections_1"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1 + T.int64(10)])
                T.writes(T_split_sections_1[v_ax0, v_ax1])
                T_split_sections_1[v_ax0, v_ax1] = x[v_ax0, v_ax1 + T.int64(10)]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_split_sections_2"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1 + T.int64(20)])
                T.writes(T_split_sections_2[v_ax0, v_ax1])
                T_split_sections_2[v_ax0, v_ax1] = x[v_ax0, v_ax1 + T.int64(20)]

    @R.function
    def main(x: R.Tensor((1, 30), dtype="float32"), w: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.split, (x,), out_sinfo=[R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32")])
            lv1: R.Tensor((1, 10), dtype="float32") = lv[0]
            gv = R.call_tir(cls.dense, (lv1, w), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(gv)
        return gv
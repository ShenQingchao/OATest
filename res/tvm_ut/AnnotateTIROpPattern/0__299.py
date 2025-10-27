# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv2: T.Buffer((T.int64(1), T.int64(10)), "float32"), lv7: T.Buffer((T.int64(1), T.int64(10)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv2[v_ax0, v_ax1], lv7[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv2[v_ax0, v_ax1] + lv7[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def dense(x: T.Buffer((T.int64(1), T.int64(10)), "float32"), w: T.Buffer((T.int64(30), T.int64(10)), "float32"), T_matmul_NT: T.Buffer((T.int64(1), T.int64(30)), "float32")):
        T.func_attr({"layout_free_buffers": [1], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(30), T.int64(10)):
            with T.block("T_matmul_NT"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], w[v_i1, v_k])
                T.writes(T_matmul_NT[v_i0, v_i1])
                with T.init():
                    T_matmul_NT[v_i0, v_i1] = T.float32(0)
                T_matmul_NT[v_i0, v_i1] = T_matmul_NT[v_i0, v_i1] + x[v_i0, v_k] * w[v_i1, v_k]

    @T.prim_func(private=True)
    def exp(lv5: T.Buffer((T.int64(1), T.int64(10)), "float32"), compute: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv5[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.exp(lv5[v_i0, v_i1])

    @T.prim_func(private=True)
    def multiply(lv4: T.Buffer((T.int64(1), T.int64(10)), "float32"), lv6: T.Buffer((T.int64(1), T.int64(10)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv4[v_ax0, v_ax1], lv6[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = lv4[v_ax0, v_ax1] * lv6[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def sigmoid(lv1: T.Buffer((T.int64(1), T.int64(10)), "float32"), compute: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv1[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(lv1[v_i0, v_i1])

    @T.prim_func(private=True)
    def split(dense: T.Buffer((T.int64(1), T.int64(30)), "float32"), T_split_sections: T.Buffer((T.int64(1), T.int64(10)), "float32"), T_split_sections_1: T.Buffer((T.int64(1), T.int64(10)), "float32"), T_split_sections_2: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_split_sections"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(dense[v_ax0, v_ax1])
                T.writes(T_split_sections[v_ax0, v_ax1])
                T_split_sections[v_ax0, v_ax1] = dense[v_ax0, v_ax1]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_split_sections_1"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(dense[v_ax0, v_ax1 + T.int64(10)])
                T.writes(T_split_sections_1[v_ax0, v_ax1])
                T_split_sections_1[v_ax0, v_ax1] = dense[v_ax0, v_ax1 + T.int64(10)]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_split_sections_2"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(dense[v_ax0, v_ax1 + T.int64(20)])
                T.writes(T_split_sections_2[v_ax0, v_ax1])
                T_split_sections_2[v_ax0, v_ax1] = dense[v_ax0, v_ax1 + T.int64(20)]

    @T.prim_func(private=True)
    def tanh(lv3: T.Buffer((T.int64(1), T.int64(10)), "float32"), compute: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv3[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.tanh(lv3[v_i0, v_i1])

    @R.function(private=True)
    def fused_split_sigmoid_tanh_exp_multiply_add(dense: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.split, (dense,), out_sinfo=[R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32")])
            lv1: R.Tensor((1, 10), dtype="float32") = lv[0]
            lv2 = R.call_tir(cls.sigmoid, (lv1,), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            lv3: R.Tensor((1, 10), dtype="float32") = lv[1]
            lv4 = R.call_tir(cls.tanh, (lv3,), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            lv5: R.Tensor((1, 10), dtype="float32") = lv[2]
            lv6 = R.call_tir(cls.exp, (lv5,), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            lv7 = R.call_tir(cls.multiply, (lv4, lv6), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            gv = R.call_tir(cls.add, (lv2, lv7), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((1, 10), dtype="float32"), w: R.Tensor((30, 10), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv8 = R.call_tir(cls.dense, (x, w), out_sinfo=R.Tensor((1, 30), dtype="float32"))
            gv1: R.Tensor((1, 10), dtype="float32") = cls.fused_split_sigmoid_tanh_exp_multiply_add(lv8)
            R.output(gv1)
        return gv1
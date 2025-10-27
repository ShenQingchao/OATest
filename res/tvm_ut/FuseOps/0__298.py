# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv3: T.Buffer((T.int64(1), T.int64(10)), "float32"), lv8: T.Buffer((T.int64(1), T.int64(10)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv3[v_ax0, v_ax1], lv8[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv3[v_ax0, v_ax1] + lv8[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def dense(x: T.Buffer((T.int64(1), T.int64(10)), "float32"), w: T.Buffer((T.int64(30), T.int64(10)), "float32"), T_matmul_NT: T.Buffer((T.int64(1), T.int64(30)), "float32")):
        T.func_attr({"layout_free_buffers": [1], "op_pattern": 4, "tir.noalias": T.bool(True)})
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
    def exp(lv6: T.Buffer((T.int64(1), T.int64(10)), "float32"), compute: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv6[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.exp(lv6[v_i0, v_i1])

    @T.prim_func(private=True)
    def multiply(lv5: T.Buffer((T.int64(1), T.int64(10)), "float32"), lv7: T.Buffer((T.int64(1), T.int64(10)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv5[v_ax0, v_ax1], lv7[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = lv5[v_ax0, v_ax1] * lv7[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def sigmoid(lv2: T.Buffer((T.int64(1), T.int64(10)), "float32"), compute: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv2[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(lv2[v_i0, v_i1])

    @T.prim_func(private=True)
    def split(lv: T.Buffer((T.int64(1), T.int64(30)), "float32"), T_split_sections: T.Buffer((T.int64(1), T.int64(10)), "float32"), T_split_sections_1: T.Buffer((T.int64(1), T.int64(10)), "float32"), T_split_sections_2: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_split_sections"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv[v_ax0, v_ax1])
                T.writes(T_split_sections[v_ax0, v_ax1])
                T_split_sections[v_ax0, v_ax1] = lv[v_ax0, v_ax1]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_split_sections_1"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv[v_ax0, v_ax1 + T.int64(10)])
                T.writes(T_split_sections_1[v_ax0, v_ax1])
                T_split_sections_1[v_ax0, v_ax1] = lv[v_ax0, v_ax1 + T.int64(10)]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_split_sections_2"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv[v_ax0, v_ax1 + T.int64(20)])
                T.writes(T_split_sections_2[v_ax0, v_ax1])
                T_split_sections_2[v_ax0, v_ax1] = lv[v_ax0, v_ax1 + T.int64(20)]

    @T.prim_func(private=True)
    def tanh(lv4: T.Buffer((T.int64(1), T.int64(10)), "float32"), compute: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv4[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.tanh(lv4[v_i0, v_i1])

    @R.function
    def main(x: R.Tensor((1, 10), dtype="float32"), w: R.Tensor((30, 10), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.dense, (x, w), out_sinfo=R.Tensor((1, 30), dtype="float32"))
            lv1 = R.call_tir(cls.split, (lv,), out_sinfo=[R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32")])
            lv2: R.Tensor((1, 10), dtype="float32") = lv1[0]
            lv3 = R.call_tir(cls.sigmoid, (lv2,), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            lv4: R.Tensor((1, 10), dtype="float32") = lv1[1]
            lv5 = R.call_tir(cls.tanh, (lv4,), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            lv6: R.Tensor((1, 10), dtype="float32") = lv1[2]
            lv7 = R.call_tir(cls.exp, (lv6,), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            lv8 = R.call_tir(cls.multiply, (lv5, lv7), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            gv = R.call_tir(cls.add, (lv3, lv8), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(gv)
        return gv
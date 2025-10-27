# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv1: T.Buffer((T.int64(1), T.int64(128)), "float32"), linear1_bias: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[v_ax0, v_ax1], linear1_bias[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv1[v_ax0, v_ax1] + linear1_bias[v_ax1]

    @T.prim_func(private=True)
    def add1(lv5: T.Buffer((T.int64(1), T.int64(10)), "float32"), linear2_bias: T.Buffer((T.int64(10),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv5[v_ax0, v_ax1], linear2_bias[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv5[v_ax0, v_ax1] + linear2_bias[v_ax1]

    @T.prim_func(private=True)
    def matmul(inp_0: T.Buffer((T.int64(1), T.int64(784)), "float32"), lv: T.Buffer((T.int64(784), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(128), T.int64(784)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(inp_0[v_i0, v_k], lv[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + inp_0[v_i0, v_k] * lv[v_k, v_i1]

    @T.prim_func(private=True)
    def matmul1(inp_1: T.Buffer((T.int64(1), T.int64(128)), "float32"), lv4: T.Buffer((T.int64(128), T.int64(10)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(10), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(inp_1[v_i0, v_k], lv4[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + inp_1[v_i0, v_k] * lv4[v_k, v_i1]

    @T.prim_func(private=True)
    def relu(lv2: T.Buffer((T.int64(1), T.int64(128)), "float32"), compute: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv2[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(lv2[v_i0, v_i1], T.float32(0))

    @T.prim_func(private=True)
    def transpose(linear1_weight: T.Buffer((T.int64(128), T.int64(784)), "float32"), T_transpose: T.Buffer((T.int64(784), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(784), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(linear1_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = linear1_weight[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose1(linear2_weight: T.Buffer((T.int64(10), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(128), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(128), T.int64(10)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(linear2_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = linear2_weight[v_ax1, v_ax0]

    @R.function
    def main(inp_0: R.Tensor((1, 784), dtype="float32"), inp_1: R.Tensor((1, 128), dtype="float32"), linear1_bias: R.Tensor((128,), dtype="float32"), linear1_weight: R.Tensor((128, 784), dtype="float32"), linear2_bias: R.Tensor((10,), dtype="float32"), linear2_weight: R.Tensor((10, 128), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.transpose, (linear1_weight,), out_sinfo=R.Tensor((784, 128), dtype="float32"))
            lv1 = R.call_tir(cls.matmul, (inp_0, lv), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv2 = R.call_tir(cls.add, (lv1, linear1_bias), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv3 = R.call_tir(cls.relu, (lv2,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv4 = R.call_tir(cls.transpose1, (linear2_weight,), out_sinfo=R.Tensor((128, 10), dtype="float32"))
            lv5 = R.call_tir(cls.matmul1, (inp_1, lv4), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            lv6 = R.call_tir(cls.add1, (lv5, linear2_bias), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return gv
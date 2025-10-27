# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(rxplaceholder: T.Buffer((T.int64(1), T.int64(128)), "float32"), rxplaceholder_1: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(rxplaceholder[v_ax0, v_ax1], rxplaceholder_1[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] + rxplaceholder_1[v_ax1]

    @T.prim_func(private=True)
    def add1(rxplaceholder: T.Buffer((T.int64(1), T.int64(10)), "float32"), rxplaceholder_1: T.Buffer((T.int64(10),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(rxplaceholder[v_ax0, v_ax1], rxplaceholder_1[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] + rxplaceholder_1[v_ax1]

    @T.prim_func(private=True)
    def matmul(rxplaceholder: T.Buffer((T.int64(1), T.int64(784)), "float32"), rxplaceholder_1: T.Buffer((T.int64(784), T.int64(128)), "float32"), matmul_1: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(128), T.int64(784)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rxplaceholder[v_i0, v_k], rxplaceholder_1[v_k, v_i1])
                T.writes(matmul_1[v_i0, v_i1])
                with T.init():
                    matmul_1[v_i0, v_i1] = T.float32(0)
                matmul_1[v_i0, v_i1] = matmul_1[v_i0, v_i1] + rxplaceholder[v_i0, v_k] * rxplaceholder_1[v_k, v_i1]

    @T.prim_func(private=True)
    def matmul1(rxplaceholder: T.Buffer((T.int64(1), T.int64(128)), "float32"), rxplaceholder_1: T.Buffer((T.int64(128), T.int64(10)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(10), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rxplaceholder[v_i0, v_k], rxplaceholder_1[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + rxplaceholder[v_i0, v_k] * rxplaceholder_1[v_k, v_i1]

    @T.prim_func(private=True)
    def relu(rxplaceholder: T.Buffer((T.int64(1), T.int64(128)), "float32"), compute: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(rxplaceholder[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(rxplaceholder[v_i0, v_i1], T.float32(0))

    @T.prim_func(private=True)
    def transpose(rxplaceholder: T.Buffer((T.int64(128), T.int64(784)), "float32"), T_transpose: T.Buffer((T.int64(784), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(784), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(rxplaceholder[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = rxplaceholder[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose1(rxplaceholder: T.Buffer((T.int64(10), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(128), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(128), T.int64(10)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(rxplaceholder[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = rxplaceholder[v_ax1, v_ax0]

    @R.function(private=True)
    def fused_matmul1_add1(inp_1: R.Tensor((1, 128), dtype="float32"), lv4: R.Tensor((128, 10), dtype="float32"), linear2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv5 = R.call_tir(cls.matmul1, (inp_1, lv4), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            gv = R.call_tir(cls.add1, (lv5, linear2_bias), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(inp_0: R.Tensor((1, 784), dtype="float32"), inp_1: R.Tensor((1, 128), dtype="float32"), linear1_bias: R.Tensor((128,), dtype="float32"), linear1_weight: R.Tensor((128, 784), dtype="float32"), linear2_bias: R.Tensor((10,), dtype="float32"), linear2_weight: R.Tensor((10, 128), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.transpose, (linear1_weight,), out_sinfo=R.Tensor((784, 128), dtype="float32"))
            lv4 = R.call_tir(cls.transpose1, (linear2_weight,), out_sinfo=R.Tensor((128, 10), dtype="float32"))
            lv_1: R.Tensor((1, 10), dtype="float32") = cls.fused_matmul1_add1(inp_1, lv4, linear2_bias)
            gv: R.Tensor((1, 10), dtype="float32") = lv_1
            R.output(gv)
        return gv
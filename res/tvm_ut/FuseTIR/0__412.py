# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(A: T.Buffer((T.int64(4096), T.int64(4096)), "float16"), Out: T.Buffer((T.int64(4096), T.int64(4096)), "float16")):
        # with T.block("root"):
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(Out[vi, vj])
                Out[vi, vj] = A[vi, vj] + T.float16(1)

    @T.prim_func(private=True)
    def take(A: T.Buffer((T.int64(4096), T.int64(4096)), "float16"), B: T.Buffer((T.int64(1),), "int32"), T_take: T.Buffer((T.int64(1), T.int64(4096)), "float16")):
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(4096)):
            with T.block("T_take"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
                T.writes(T_take[v_ax0, v_ax1])
                T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

    @R.function(private=True)
    def fused_func(input_ids: R.Tensor((1,), dtype="int32"), input_embeds: R.Tensor((4096, 4096), dtype="float16")) -> R.Tensor((1, 4096), dtype="float16"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.add, (input_embeds,), out_sinfo=R.Tensor((4096, 4096), dtype="float16"))
            gv = R.call_tir(cls.take, (lv, input_ids), out_sinfo=R.Tensor((1, 4096), dtype="float16"))
            R.output(gv)
        return gv

    @R.function
    def main(input_ids: R.Tensor((1,), dtype="int32"), input_embeds: R.Tensor((4096, 4096), dtype="float16")) -> R.Tensor((1, 4096), dtype="float16"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((1, 4096), dtype="float16") = cls.fused_func(input_ids, input_embeds)
            R.output(gv)
        return gv
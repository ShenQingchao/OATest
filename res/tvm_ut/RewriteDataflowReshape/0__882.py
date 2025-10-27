# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def add_one(A: T.Buffer((T.int64(1), T.int64(1000)), "int32"), T_add_one: T.Buffer((T.int64(1), T.int64(1000)), "int32")):
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(1000)):
            with T.block("T_add_one"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1])
                T.writes(T_add_one[v_ax0, v_ax1])
                T_add_one[v_ax0, v_ax1] = A[v_ax0, v_ax1] + 1

    @T.prim_func
    def strided_slice(A: T.Buffer((T.int64(1), T.int64(1024)), "int32"), T_strided_slice: T.Buffer((T.int64(1), T.int64(1000)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(1000)):
            with T.block("T_strided_slice"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1])
                T.writes(T_strided_slice[v_ax0, v_ax1])
                T_strided_slice[v_ax0, v_ax1] = A[v_ax0, v_ax1]

    @R.function
    def main(A: R.Tensor((1, 1024), dtype="int32")) -> R.Tensor((1, 1000), dtype="int32"):
        cls = Module
        with R.dataflow():
            S = R.call_tir(cls.strided_slice, (A,), out_sinfo=R.Tensor((1, 1000), dtype="int32"))
            A_1 = R.call_tir(cls.add_one, (S,), out_sinfo=R.Tensor((1, 1000), dtype="int32"))
            R.output(A_1)
        return A_1
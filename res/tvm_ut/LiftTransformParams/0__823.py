# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def zeros(var_T_full: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        T_full = T.match_buffer(var_T_full, (n, n))
        # with T.block("root"):
        for ax0, ax1 in T.grid(n, n):
            with T.block("T_full"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads()
                T.writes(T_full[v_ax0, v_ax1])
                T_full[v_ax0, v_ax1] = T.float32(0)

    @R.function
    def main(shape: R.Shape(["n"])) -> R.Shape(["n"]):
        n = T.int64()
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            zeros = R.call_tir(cls.zeros, R.tuple(), out_sinfo=R.Tensor((n, n), dtype="float32"))
            R.output()
        return shape
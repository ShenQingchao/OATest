# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(A: T.Buffer((T.int64(10), T.int64(20)), "float32"), B: T.Buffer((), "float32"), Out: T.Buffer((T.int64(10), T.int64(20)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1], B[()])
                T.writes(Out[v_ax0, v_ax1])
                Out[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[()]

    @R.function(private=True)
    def fused_sums(x: R.Tensor((10, 20), dtype="float32"), p0: R.Tensor((), dtype="float32")) -> R.Tensor((10, 20), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.add, (x, p0), out_sinfo=R.Tensor((10, 20), dtype="float32"))
            lv1: R.Tensor((10, 20), dtype="float32") = R.call_tir_inplace(cls.add, (x, p0, lv), out_sinfo=R.Tensor((10, 20), dtype="float32"), inplace_indices=[2])
            lv2: R.Tensor((10, 20), dtype="float32") = R.call_tir_inplace(cls.add, (x, p0, lv1), out_sinfo=R.Tensor((10, 20), dtype="float32"), inplace_indices=[2])
            R.output(lv2)
        return lv2

    @R.function
    def main(x: R.Tensor((10, 20), dtype="float32"), p0: R.Tensor((), dtype="float32")) -> R.Tensor((10, 20), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv1: R.Tensor((10, 20), dtype="float32") = cls.fused_sums(x, p0)
            R.output(gv1)
        return gv1
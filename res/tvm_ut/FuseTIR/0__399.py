# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def dynamic_tir_kernel(a: T.handle, b: T.handle):
        m, n = T.int64(), T.int64()
        A = T.match_buffer(a, (m, n))
        B = T.match_buffer(b, (m, n))
        # with T.block("root"):
        for iters_0, iters_1 in T.grid(m, n):
            with T.block("compute"):
                i, j = T.axis.remap("SS", [iters_0, iters_1])
                T.reads(A[i, j])
                T.writes(B[i, j])
                B[i, j] = A[i, j] * T.Cast("float32", i) + T.Cast("float32", j)

    @R.function(private=True)
    def fused_function(x: R.Tensor((16, 32), dtype="float32")) -> R.Tensor((16, 32), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            y = R.call_tir(cls.dynamic_tir_kernel, (x,), out_sinfo=R.Tensor((16, 32), dtype="float32"))
            z = R.call_tir(cls.dynamic_tir_kernel, (y,), out_sinfo=R.Tensor((16, 32), dtype="float32"))
            R.output(z)
        return z

    @R.function
    def main(x: R.Tensor((16, 32), dtype="float32")) -> R.Tensor((16, 32), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((16, 32), dtype="float32") = cls.fused_function(x)
            R.output(gv)
        return gv
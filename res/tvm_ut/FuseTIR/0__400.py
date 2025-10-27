# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def dynamic_tir_kernel(a: T.handle, b: T.handle, c: T.handle, d: T.handle):
        m = T.int64()
        n = T.int64()
        A = T.match_buffer(a, (m * n,))
        B = T.match_buffer(b, (m,))
        C = T.match_buffer(c, (n,))
        D = T.match_buffer(d, (m * n,))
        # with T.block("root"):
        for i, j in T.grid(m, n):
            with T.block("compute"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi * T.int64(32) + vj], B[vi], C[vj])
                T.writes(D[vi * T.int64(32) + vj])
                D[vi * T.int64(32) + vj] = A[vi * T.int64(32) + vj] * B[vi] + C[vj]

    @R.function(private=True)
    def fused_function(x: R.Tensor((512,), dtype="float32"), B: R.Tensor((16,), dtype="float32"), C: R.Tensor((32,), dtype="float32")) -> R.Tensor((512,), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            y = R.call_tir(cls.dynamic_tir_kernel, (x, B, C), out_sinfo=R.Tensor((512,), dtype="float32"))
            z = R.call_tir(cls.dynamic_tir_kernel, (y, B, C), out_sinfo=R.Tensor((512,), dtype="float32"))
            R.output(z)
        return z

    @R.function
    def main(x: R.Tensor((512,), dtype="float32"), B: R.Tensor((16,), dtype="float32"), C: R.Tensor((32,), dtype="float32")) -> R.Tensor((512,), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((512,), dtype="float32") = cls.fused_function(x, B, C)
            R.output(gv)
        return gv
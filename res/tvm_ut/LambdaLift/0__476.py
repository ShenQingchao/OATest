# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def sub(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")):
        # with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("sub"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj], B[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = A[vi, vj] - B[vi, vj]

    @R.function
    def before(c0: R.Tensor((16, 16), dtype="float32"), x: R.Tensor(dtype="float32", ndim=2)) -> R.Tensor((16, 16), dtype="float32"):
        cls = Module
        s = R.call_tir(cls.sub, (c0, x), out_sinfo=R.Tensor((16, 16), dtype="float32"))
        return s
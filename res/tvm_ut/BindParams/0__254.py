# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def identity(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32")):
        # with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("identity"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj]

    @R.function
    def main(c0: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv0 = R.call_tir(cls.identity, (c0,), out_sinfo=R.Tensor((16, 16), dtype="float32"))
            R.output(gv0)
        return gv0
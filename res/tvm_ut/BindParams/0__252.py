# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def addone(A: T.Buffer((2, 2), "float32"), B: T.Buffer((2, 2), "float32")):
        # with T.block("root"):
        for i, j in T.grid(2, 2):
            with T.block("addone"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] + T.float32(1)

    @R.function
    def main(c1: R.Tensor((2, 2), dtype="float32"), c2: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
        return c2
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(A: R.Tensor(("M", "N")), B: R.Tensor(("N", "P")), C: R.Tensor(("P", "Q"))) -> R.Tensor(("M", "Q")):
        M = T.int64()
        Q = T.int64()
        N = T.int64()
        P = T.int64()
        gv: R.Tensor((M, P)) = R.matmul(A, B, out_dtype="void")
        out: R.Tensor((M, Q)) = R.matmul(gv, C, out_dtype="void")
        return out
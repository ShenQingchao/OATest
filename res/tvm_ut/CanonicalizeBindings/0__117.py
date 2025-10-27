# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def main(A: R.Tensor(("n", 16), dtype="int32")) -> R.Tensor(("n", 16), dtype="int32"):
        n = T.int64()
        B: R.Tensor((n, 16), dtype="int32") = A
        C: R.Tensor(dtype="int32", ndim=2) = R.add(B, B)
        return C
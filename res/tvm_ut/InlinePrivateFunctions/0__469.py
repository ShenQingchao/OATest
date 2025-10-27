# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def subroutine(B: R.Tensor((16, 16), dtype="int32")) -> R.Tensor((16, 32), dtype="int32"):
        C: R.Tensor((16, 32), dtype="int32") = R.concat((B, B), axis=1)
        return C

    @R.function
    def main(A: R.Tensor((16, 16), dtype="int32")) -> R.Tensor((16, 32), dtype="int32"):
        cls = Module
        B: R.Tensor((16, 16), dtype="int32") = R.multiply(A, A)
        C: R.Tensor((16, 32), dtype="int32") = cls.subroutine(B)
        D: R.Tensor((16, 32), dtype="int32") = R.add(C, C)
        return D
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(A: R.Tensor((16, 32, 64), dtype="float32"), Weight: R.Tensor((64, 128), dtype="float32"), Bias: R.Tensor((16, 32, 128), dtype="float32")) -> R.Tensor((16, 32, 128), dtype="float32"):
        gv: R.Tensor((16, 32, 128), dtype="float32") = R.matmul(A,Weight)
        gv = R.add(gv, Bias)
        return gv

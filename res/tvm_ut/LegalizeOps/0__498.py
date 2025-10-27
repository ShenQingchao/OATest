# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(A: R.Tensor((16, 32)), B: R.Tensor((32, 8))) -> R.Tensor((16, 8)):
        gv: R.Tensor((16, 8)) = R.matmul(A, B, out_dtype="void")
        return gv
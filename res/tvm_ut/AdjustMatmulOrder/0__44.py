# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,)), A: R.Tensor((32, 2)), B: R.Tensor((2, 16))) -> R.Tensor((32,)):
        x_1: R.Tensor((2,)) = R.matmul(B, x, out_dtype="void")
        x_2: R.Tensor((32,)) = R.matmul(A, x_1, out_dtype="void")
        return x_2
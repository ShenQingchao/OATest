# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,)), A: R.Tensor((16, 2)), B: R.Tensor((2, 32))) -> R.Tensor((32,)):
        x_1: R.Tensor((2,)) = R.matmul(x, A, out_dtype="void")
        x_2: R.Tensor((32,)) = R.matmul(x_1, B, out_dtype="void")
        return x_2
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,)), A: R.Tensor((32, 2)), B: R.Tensor((2, 16))) -> R.Tensor((32,)):
        weight: R.Tensor((32, 16)) = R.matmul(A, B, out_dtype="void")
        out: R.Tensor((32,)) = R.matmul(weight, x, out_dtype="void")
        return out
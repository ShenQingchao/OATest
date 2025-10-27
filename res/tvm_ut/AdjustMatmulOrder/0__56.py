# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,)), A: R.Tensor((32, 2)), B: R.Tensor((2, 16))) -> R.Tensor((32,)):
        linear_weight: R.Tensor((32, 16)) = R.matmul(A, B, out_dtype="void")
        matmul_weight: R.Tensor((16, 32)) = R.permute_dims(linear_weight, axes=None)
        out: R.Tensor((32,)) = R.matmul(x, matmul_weight, out_dtype="void")
        return out
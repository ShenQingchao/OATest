# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,), dtype="float32"), A: R.Tensor((32, 16), dtype="float32"), B: R.Tensor((32, 16), dtype="float32")) -> R.Tensor((32,), dtype="float32"):
        linear_weight: R.Tensor((32, 16), dtype="float32") = R.add(A, B)
        matmul_weight: R.Tensor((16, 32), dtype="float32") = R.permute_dims(linear_weight, axes=None)
        out: R.Tensor((32,), dtype="float32") = R.matmul(x, matmul_weight, out_dtype="void")
        return out
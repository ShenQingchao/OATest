# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,), dtype="float32"), weight: R.Tensor((32, 16), dtype="float32")) -> R.Tensor((32,), dtype="float32"):
        out: R.Tensor((32,), dtype="float32") = R.matmul(weight, x, out_dtype="void")
        return out
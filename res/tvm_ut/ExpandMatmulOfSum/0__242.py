# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((16,), dtype="float32"), A: R.Tensor((16, 32), dtype="float32"), B: R.Tensor((16, 32), dtype="float32")) -> R.Tensor((32,), dtype="float32"):
        R.func_attr({"num_input": 1})
        weight: R.Tensor((16, 32), dtype="float32") = R.add(A, B)
        out: R.Tensor((32,), dtype="float32") = R.matmul(x, weight, out_dtype="void")
        return out
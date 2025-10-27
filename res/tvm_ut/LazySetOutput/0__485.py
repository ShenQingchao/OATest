# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def transform_params(A: R.Tensor((16, 16), dtype="float32"), B: R.Tensor((16, 16), dtype="float32")) -> R.Tuple(R.Tensor((16, 16), dtype="float32"), R.Tensor((16, 16), dtype="float32")):
        C: R.Tensor((16, 16), dtype="float32") = R.multiply(A, R.const(2, "float32"))
        D: R.Tensor((16, 16), dtype="float32") = R.add(C, B)
        return (D, C)
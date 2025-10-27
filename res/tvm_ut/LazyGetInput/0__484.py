# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def transform_params(A: R.Tensor(("m", "n"), dtype="float32"), B: R.Tensor(("m", "n"), dtype="float32")) -> R.Tuple(R.Tensor(("m", "n"), dtype="float32"), R.Tensor(("m", "n"), dtype="float32")):
        m = T.int64()
        n = T.int64()
        C: R.Tensor((m, n), dtype="float32") = R.multiply(A, R.const(2, "float32"))
        D: R.Tensor((m, n), dtype="float32") = R.add(C, B)
        return (D, B)
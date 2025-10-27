# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, "n", 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
        n = T.int64()
        gv: R.Tensor((2, 4), dtype="float32") = R.take(x, R.prim_value(n - 1), axis=1)
        return gv
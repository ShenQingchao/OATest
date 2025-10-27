# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
        m = T.int64()
        n = T.int64()
        gv: R.Tensor((m, n), dtype="float32") = R.cos(x)
        return gv
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(condition: R.Tensor(("a", "b", 1), dtype="bool"), x: R.Tensor(("b", "c"), dtype="float32"), y: R.Tensor(("b", 1), dtype="float32")) -> R.Tensor(("a", "b", "c"), dtype="float32"):
        a = T.int64()
        b = T.int64()
        c = T.int64()
        gv: R.Tensor((a, b, c), dtype="float32") = R.where(condition, x, y)
        return gv
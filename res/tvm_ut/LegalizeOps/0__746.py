# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("a", "b", "c", "d"), dtype="float32")) -> R.Tensor((1, "b", "c", 1), dtype="float32"):
        b = T.int64()
        c = T.int64()
        a = T.int64()
        d = T.int64()
        gv: R.Tensor((1, b, c, 1), dtype="float32") = R.variance(x, axis=[0, 3], keepdims=True)
        return gv
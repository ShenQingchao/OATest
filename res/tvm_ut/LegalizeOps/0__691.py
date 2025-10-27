# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("a", "b", "c"), dtype="float32")) -> R.Tensor(("a", "b", "c"), dtype="float32"):
        a = T.int64()
        b = T.int64()
        c = T.int64()
        gv: R.Tensor((a, b, c), dtype="float32") = R.nn.softmax(x, axis=-1)
        return gv
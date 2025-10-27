# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("m", "n"), dtype="float32"), indices: R.Tensor(("i",), dtype="int64")) -> R.Tensor(("m", "i"), dtype="float32"):
        m = T.int64()
        i = T.int64()
        n = T.int64()
        gv: R.Tensor((m, i), dtype="float32") = R.take(x, indices, axis=1)
        return gv
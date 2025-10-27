# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("a", "b", "c", "d"), dtype="float32")) -> R.Tensor(("b", "d", "c", "a"), dtype="float32"):
        b = T.int64()
        d = T.int64()
        c = T.int64()
        a = T.int64()
        gv: R.Tensor((b, d, c, a), dtype="float32") = R.permute_dims(x, axes=[1, -1, 2, -4])
        return gv
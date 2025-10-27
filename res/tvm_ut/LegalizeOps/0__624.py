# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(t: R.Tuple(R.Tensor(("a", "b0"), dtype="float32"), R.Tensor(("a", "b1"), dtype="float32"), R.Tensor(("a", "b2"), dtype="float32"))) -> R.Tensor(("a", "b0 + b1 + b2"), dtype="float32"):
        a = T.int64()
        b0 = T.int64()
        b1 = T.int64()
        b2 = T.int64()
        gv: R.Tensor((a, b0 + b1 + b2), dtype="float32") = R.concat(t, axis=1)
        return gv
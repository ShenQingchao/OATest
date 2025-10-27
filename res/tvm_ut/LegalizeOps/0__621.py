# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(dumb_param: R.Tensor(("a", "c")), x: R.Tensor(("b", 1, "d"), dtype="float32")) -> R.Tensor(("a", "b", "c", "d"), dtype="float32"):
        a = T.int64()
        b = T.int64()
        c = T.int64()
        d = T.int64()
        gv: R.Tensor((a, b, c, d), dtype="float32") = R.broadcast_to(x, R.shape([a, b, c, d]))
        return gv
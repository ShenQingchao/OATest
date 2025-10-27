# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(dumb_param: R.Tensor(("m", "n")), v: R.Tensor((), dtype="int32")) -> R.Tensor(("m", "n"), dtype="int32"):
        m = T.int64()
        n = T.int64()
        gv: R.Tensor((m, n), dtype="int32") = R.full(R.shape([m, n]), v, dtype="int32")
        return gv
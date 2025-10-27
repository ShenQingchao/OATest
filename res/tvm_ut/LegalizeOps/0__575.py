# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(dumb_param: R.Tensor(("m", "n"))) -> R.Tensor(("m", "n"), dtype="float32"):
        m = T.int64()
        n = T.int64()
        gv: R.Tensor((m, n), dtype="float32") = R.ones(R.shape([m, n]), dtype="float32")
        return gv
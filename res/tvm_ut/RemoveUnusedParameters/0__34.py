# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def func(A: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
        m = T.int64()
        n = T.int64()
        gv: R.Tensor((m, n), dtype="float32") = R.zeros(R.shape([m, n]), dtype="float32")
        return gv

    @R.function
    def main(A: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
        m = T.int64()
        n = T.int64()
        cls = Module
        gv: R.Tensor((m, n), dtype="float32") = cls.func(A)
        return gv
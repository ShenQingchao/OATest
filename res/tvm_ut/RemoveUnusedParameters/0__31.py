# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def func(A: R.Tensor(("m", "n"), dtype="float32"), _m: R.Prim(value="m"), _n: R.Prim(value="n")) -> R.Tensor(("m", "n"), dtype="float32"):
        m = T.int64()
        n = T.int64()
        zeros: R.Tensor((m, n), dtype="float32") = R.zeros(R.shape([m, n]), dtype="float32")
        out: R.Tensor((m, n), dtype="float32") = R.add(A, zeros)
        return out

    @R.function
    def main(A: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
        m = T.int64()
        n = T.int64()
        cls = Module
        gv: R.Tensor((m, n), dtype="float32") = cls.func(A, R.prim_value(m), R.prim_value(n))
        return gv
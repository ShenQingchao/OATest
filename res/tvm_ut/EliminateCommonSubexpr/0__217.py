# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor(dtype="float32"), y: R.Tensor(dtype="float32")) -> R.Tensor(dtype="float32", ndim=2):
        n = T.int64()
        m = T.int64()
        p = T.int64()
        q = T.int64()
        with R.dataflow():
            A1: R.Tensor(dtype="float32") = R.add(x, y)
            B1: R.Tensor((n, m), dtype="float32") = R.match_cast(A1, R.Tensor((n, m), dtype="float32"))
            A2: R.Tensor(dtype="float32") = R.add(x, y)
            B2: R.Tensor((p, q), dtype="float32") = R.match_cast(A2, R.Tensor((p, q), dtype="float32"))
            gv: R.Tensor(dtype="float32", ndim=2) = R.multiply(B1, B2)
            R.output(gv)
        return gv
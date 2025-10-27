# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        with R.dataflow():
            A1: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            B1: R.Tensor((2, 3), dtype="float32") = R.match_cast(A1, R.Tensor((2, 3), dtype="float32"))
            A2: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            B2: R.Tensor((2, 3), dtype="float32") = R.match_cast(A2, R.Tensor((2, 3), dtype="float32"))
            gv: R.Tensor((2, 3), dtype="float32") = R.multiply(B1, B2)
            R.output(gv)
        return gv
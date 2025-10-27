# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="int32"), v: R.Tensor((), dtype="float32")) -> R.Tensor((2, 3), dtype="float64"):
        gv: R.Tensor((2, 3), dtype="float64") = R.full_like(x, v, dtype="float64")
        return gv
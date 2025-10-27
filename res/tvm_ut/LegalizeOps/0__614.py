# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((4,), dtype="float32"), y: R.Tensor((4,), dtype="float32")) -> R.Tensor((), dtype="float32"):
        gv: R.Tensor((), dtype="float32") = R.matmul(x, y, out_dtype="void")
        return gv
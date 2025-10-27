# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 1, 4, 5), dtype="float32"), y: R.Tensor((1, 1, 5, 7), dtype="float32")) -> R.Tensor((1, 1, 4, 7), dtype="float32"):
        gv: R.Tensor((1, 1, 4, 7), dtype="float32") = R.matmul(x, y, out_dtype="float32")
        return gv
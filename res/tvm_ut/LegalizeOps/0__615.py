# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 4, 5), dtype="float16"), y: R.Tensor((6, 2, 3, 5, 7), dtype="float16")) -> R.Tensor((6, 2, 3, 4, 7), dtype="float32"):
        gv: R.Tensor((6, 2, 3, 4, 7), dtype="float32") = R.matmul(x, y, out_dtype="float32")
        return gv
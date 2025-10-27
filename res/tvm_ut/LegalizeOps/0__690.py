# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 16, 32), dtype="float32")) -> R.Tensor((2, 3, 16, 32), dtype="float32"):
        gv: R.Tensor((2, 3, 16, 32), dtype="float32") = R.nn.softmax(x, axis=-2)
        return gv
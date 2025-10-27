# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tensor((2, 4, 5), dtype="int64"):
        gv: R.Tensor((2, 4, 5), dtype="int64") = R.argmax(x, axis=1, keepdims=False)
        return gv
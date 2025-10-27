# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((8, 3), dtype="float32"):
        lv: R.Shape([8, 3]) = R.shape([8, 3])
        gv: R.Tensor((8, 3), dtype="float32") = R.reshape(x, lv)
        return gv
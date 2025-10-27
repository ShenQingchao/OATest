# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((10, "b"), dtype="float32")) -> R.Tensor((5, "b * 2"), dtype="float32"):
        b = T.int64()
        lv: R.Shape([5, b * 2]) = R.shape([5, b * 2])
        gv: R.Tensor((5, b * 2), dtype="float32") = R.reshape(x, lv)
        return gv
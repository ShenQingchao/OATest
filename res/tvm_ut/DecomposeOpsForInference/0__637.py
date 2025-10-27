# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3,), dtype="int64")) -> R.Tensor((3,), dtype="int64"):
        lv: R.Shape([3]) = R.tensor_to_shape(x)
        gv: R.Tensor((3,), dtype="int64") = R.reshape(x, lv)
        return gv
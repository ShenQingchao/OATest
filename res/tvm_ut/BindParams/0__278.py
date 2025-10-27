# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((5, 4, 3, 2), dtype="float32"), new_shape: R.Tensor((1, 1), dtype="int64")) -> R.Tensor((1, 1), dtype="int64"):
        return new_shape
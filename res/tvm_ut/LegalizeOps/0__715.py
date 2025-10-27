# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 128, 28), dtype="float32")) -> R.Tensor((2, 130, 30), dtype="float32"):
        gv: R.Tensor((2, 130, 30), dtype="float32") = R.nn.pad(x, pad_width=[0, 0, 1, 1, 1, 1], pad_mode="constant")
        return gv

# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 4, 5), dtype="float16"), weight: R.Tensor((4, 5), dtype="float16")) -> R.Tensor((2, 3, 4, 5), dtype="float16"):
        gv: R.Tensor((2, 3, 4, 5), dtype="float16") = R.nn.rms_norm(x, weight, axes=[-2, -1], epsilon=1.0000000000000001e-05)
        return gv
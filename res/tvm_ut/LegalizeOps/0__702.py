# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 4, 4, 5), dtype="float32"), gamma: R.Tensor((4,), dtype="float32"), beta: R.Tensor((4,), dtype="float32")) -> R.Tensor((2, 4, 4, 5), dtype="float32"):
        gv: R.Tensor((2, 4, 4, 5), dtype="float32") = R.nn.group_norm(x, gamma, beta, num_groups=2, channel_axis=1, axes=[2, 3], epsilon=1.0000000000000001e-05, center=True, scale=True)
        return gv
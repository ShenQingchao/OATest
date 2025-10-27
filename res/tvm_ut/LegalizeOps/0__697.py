# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), gamma: R.Tensor((3,), dtype="float32"), beta: R.Tensor((3,), dtype="float32"), moving_mean: R.Tensor((3,), dtype="float32"), moving_var: R.Tensor((3,), dtype="float32")) -> R.Tuple(R.Tensor((2, 3, 28, 28), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")):
        gv: R.Tuple(R.Tensor((2, 3, 28, 28), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
        return gv
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32"), gamma: R.Tensor((4,), dtype="float32"), beta: R.Tensor((4,), dtype="float32"), moving_mean: R.Tensor((4,), dtype="float32"), moving_var: R.Tensor((4,), dtype="float32")) -> R.Tuple(R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((4,), dtype="float32"), R.Tensor((4,), dtype="float32")):
        with R.dataflow():
            gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(x, w, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            gv2: R.Tuple(R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((4,), dtype="float32"), R.Tensor((4,), dtype="float32")) = R.nn.batch_norm(gv, gamma, beta, moving_mean, moving_var, axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001)
            R.output(gv2)
        return gv2
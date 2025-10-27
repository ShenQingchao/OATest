# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((1, 64, 56, 56), dtype="float32"), weight: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        with R.dataflow():
            conv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(data, weight, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            relu1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(conv1)
            gelu1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.gelu(conv1)
            out: R.Tensor((1, 64, 54, 54), dtype="float32") = R.add(relu1, gelu1)
            R.output(out)
        return out
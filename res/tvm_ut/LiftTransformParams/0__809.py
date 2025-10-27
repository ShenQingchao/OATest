# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 16, 224, 224), dtype="float32"), w1: R.Tensor((16, 16, 3, 3), dtype="float32"), w2: R.Tensor((16, 16, 3, 3), dtype="float32"), cond: R.Tensor((), dtype="bool")) -> R.Tensor((1, 16, 224, 224), dtype="float32"):
        R.func_attr({"num_input": 1})
        if cond:
            w: R.Tensor((16, 16, 3, 3), dtype="float32") = w1
        else:
            w: R.Tensor((16, 16, 3, 3), dtype="float32") = w2
        with R.dataflow():
            conv1: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(x, w, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            R.output(conv1)
        return conv1
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 16, 224, 224), dtype="float32"), w1: R.Tensor((16, 16, 3, 3), dtype="float32")) -> R.Tensor((1, 16, 224, 224), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            l0: R.Tuple(R.Tensor((16, 16, 3, 3), dtype="float32")) = (w1,)
            l1: R.Tuple(R.Tuple(R.Tensor((16, 16, 3, 3), dtype="float32"))) = (l0,)
            l2: R.Tuple(R.Tensor((16, 16, 3, 3), dtype="float32")) = l1[0]
            l3: R.Tensor((16, 16, 3, 3), dtype="float32") = l2[0]
            conv1: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(x, l3, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            conv2: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(conv1, w1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            R.output(conv2)
        return conv2
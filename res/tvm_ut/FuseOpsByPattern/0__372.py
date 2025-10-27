# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main2(data: R.Tensor((16, 32, 32, 16), dtype="float16"), weight1: R.Tensor((16, 3, 3, 16), dtype="float16"), weight2: R.Tensor((16, 3, 3, 16), dtype="float16")) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        with R.dataflow():
            conv1: R.Tensor((16, 32, 32, 16), dtype="float16") = R.nn.conv2d(data, weight1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="void")
            conv2: R.Tensor((16, 32, 32, 16), dtype="float16") = R.nn.conv2d(conv1, weight2, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="void")
            R.output(conv2)
        return conv2

    @R.function
    def main(data: R.Tensor((16, 32, 32, 16), dtype="float16"), weight1: R.Tensor((16, 3, 3, 16), dtype="float16"), weight2: R.Tensor((16, 3, 3, 16), dtype="float16")) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        cls = Module
        with R.dataflow():
            conv1: R.Tensor((16, 32, 32, 16), dtype="float16") = R.nn.conv2d(data, weight1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="void")
            conv2: R.Tensor((16, 32, 32, 16), dtype="float16") = R.nn.conv2d(conv1, weight2, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="void")
            conv3: R.Tensor((16, 32, 32, 16), dtype="float16") = cls.main2(data, weight1, weight2)
            result: R.Tensor((16, 32, 32, 16), dtype="float16") = R.add(conv2, conv3)
            R.output(result)
        return result
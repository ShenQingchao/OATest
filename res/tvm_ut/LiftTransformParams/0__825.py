# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 16, 224, "n"), dtype="float32"), w1: R.Tensor((16, "m", 3, 3), dtype="float32"), w2: R.Tensor((16, "m", 3, 3), dtype="float32")) -> R.Tensor((1, 16, 224, "n"), dtype="float32"):
        n = T.int64()
        m = T.int64()
        R.func_attr({"num_input": 1})
        with R.dataflow():
            zeros: R.Tensor((n, n), dtype="float32") = R.zeros(R.shape([n, n]), dtype="float32")
            w1_1: R.Tensor((16, m, 3, 3), dtype="float32") = R.add(w1, R.const(1, "float32"))
            conv1: R.Tensor((1, 16, 224, n), dtype="float32") = R.nn.conv2d(x, w1_1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            conv2: R.Tensor((1, 16, 224, n), dtype="float32") = R.nn.conv2d(conv1, w2, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            R.output(conv2)
        return conv2
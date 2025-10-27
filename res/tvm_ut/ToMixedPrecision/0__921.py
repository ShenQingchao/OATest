# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32"), w_2: R.Tensor((4, 4, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(x, w, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(x, w, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            gv3: R.Tuple(R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((2, 4, 26, 26), dtype="float32")) = gv, gv2
            gv4: R.Tuple(R.Tuple(R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((2, 4, 26, 26), dtype="float32")), R.Tensor((2, 4, 26, 26), dtype="float32")) = gv3, gv2
            gv5: R.Tuple(R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((2, 4, 26, 26), dtype="float32")) = gv4[0]
            gv6: R.Tensor((2, 4, 26, 26), dtype="float32") = gv5[0]
            gv7: R.Tensor((2, 4, 24, 24), dtype="float32") = R.nn.conv2d(gv6, w_2, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            R.output(gv7)
        return gv7
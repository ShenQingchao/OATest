# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 13, 13), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(x, w, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            gv2: R.Tensor((2, 4, 13, 13), dtype="float32") = R.nn.max_pool2d(gv, pool_size=[2, 2], strides=[2, 2], dilation=[1, 1], padding=[0, 0, 0, 0], ceil_mode=False, count_include_pad=False, layout="NCHW", out_layout="NCHW")
            R.output(gv2)
        return gv2
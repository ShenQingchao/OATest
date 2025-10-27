# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 13, 13), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
            lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
            gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(lv, lv1, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="float32")
            lv2: R.Tensor((2, 13, 13, 4), dtype="float32") = R.nn.max_pool2d(gv, pool_size=[2, 2], strides=[2, 2], dilation=[1, 1], padding=[0, 0, 0, 0], ceil_mode=False, count_include_pad=False, layout="NHWC", out_layout="NHWC")
            gv2: R.Tensor((2, 4, 13, 13), dtype="float32") = R.permute_dims(lv2, axes=[0, 3, 1, 2])
            R.output(gv2)
        return gv2
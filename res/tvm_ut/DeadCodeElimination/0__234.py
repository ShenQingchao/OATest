# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")) -> R.Tensor((2, 26, 26, 4), dtype="float16"):
        with R.dataflow():
            lv0: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
            R.output(lv0)
        gv_x: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
        gv_w: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
        with R.dataflow():
            lv1: R.Tensor((2, 28, 28, 3), dtype="float16") = R.permute_dims(gv_x, axes=[0, 2, 3, 1])
            lv2: R.Tensor((4, 3, 3, 3), dtype="float16") = R.permute_dims(gv_w, axes=[0, 2, 3, 1])
            lv3: R.Tensor((2, 26, 26, 4), dtype="float16") = R.nn.conv2d(lv1, lv2, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="void")
            lv4: R.Tensor((2, 3, 28, 28), dtype="float32") = R.permute_dims(lv0, axes=[0, 3, 1, 2])
            R.output(lv3)
        return lv3
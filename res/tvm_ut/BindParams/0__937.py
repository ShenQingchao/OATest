# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 4, 64, 64), dtype="float32"), w: R.Tensor((512, 4, 3, 3), dtype="float32"), bias: R.Tensor((512,), dtype="float32")) -> R.Tensor((1, 512, 62, 62), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((1, 4, 64, 64), dtype="float16") = R.astype(x, dtype="float16")
            lv1: R.Tensor((512, 4, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
            lv142: R.Tensor((1, 512, 62, 62), dtype="float16") = R.nn.conv2d(lv, lv1, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float16")
            lv143: R.Tensor((1, 512, 1, 1), dtype="float32") = R.reshape(bias, R.shape([1, 512, 1, 1]))
            lv2: R.Tensor((1, 512, 62, 62), dtype="float32") = R.astype(lv142, dtype="float32")
            lv144: R.Tensor((1, 512, 62, 62), dtype="float32") = R.add(lv2, lv143)
            R.output(lv144)
        return lv144
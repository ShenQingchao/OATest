# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(z: R.Tensor((1, 4, 64, 64), dtype="float32"), w0: R.Tensor((512, 4, 3, 3), dtype="float16"), w1: R.Tensor((512,), dtype="float16"), w2: R.Tensor((4, 4, 1, 1), dtype="float16"), w3: R.Tensor((4,), dtype="float16")) -> R.Tensor((1, 512, 64, 64), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((1, 4, 64, 64), dtype="float16") = R.astype(z, dtype="float16")
            lv_1: R.Tensor((512, 4, 3, 3), dtype="float16") = w0
            lv1: R.Tensor((512,), dtype="float16") = w1
            lv140: R.Tensor((4, 4, 1, 1), dtype="float16") = w2
            lv141: R.Tensor((4,), dtype="float16") = w3
            lv1_1: R.Tensor((1, 4, 64, 64), dtype="float32") = R.nn.conv2d(lv, lv140, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv142: R.Tensor((1, 4, 64, 64), dtype="float16") = R.astype(lv1_1, dtype="float16")
            lv143: R.Tensor((1, 4, 1, 1), dtype="float16") = R.reshape(lv141, R.shape([1, 4, 1, 1]))
            lv144: R.Tensor((1, 4, 64, 64), dtype="float16") = R.add(lv142, lv143)
            lv2: R.Tensor((1, 512, 64, 64), dtype="float32") = R.nn.conv2d(lv144, lv_1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv145: R.Tensor((1, 512, 64, 64), dtype="float16") = R.astype(lv2, dtype="float16")
            lv146: R.Tensor((1, 512, 1, 1), dtype="float16") = R.reshape(lv1, R.shape([1, 512, 1, 1]))
            lv147: R.Tensor((1, 512, 64, 64), dtype="float16") = R.add(lv145, lv146)
            gv: R.Tensor((1, 512, 64, 64), dtype="float32") = R.astype(lv147, dtype="float32")
            R.output(gv)
        return gv
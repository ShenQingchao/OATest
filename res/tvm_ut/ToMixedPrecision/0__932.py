# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 4, 64, 64), dtype="float32"), w: R.Tensor((512, 4, 3, 3), dtype="float32"), bias: R.Tensor((512, 1, 1), dtype="float32")) -> R.Tensor((1, 256, 64, 64), dtype="float32"):
        with R.dataflow():
            conv: R.Tensor((1, 512, 63, 63), dtype="float32") = R.nn.conv2d(x, w, strides=[1, 1], padding=[0, 0, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            bias_out: R.Tensor((1, 512, 63, 63), dtype="float32") = R.add(conv, bias)
            split: R.Tuple(R.Tensor((1, 256, 63, 63), dtype="float32"), R.Tensor((1, 256, 63, 63), dtype="float32")) = R.split(bias_out, indices_or_sections=2, axis=1)
            lv3: R.Tensor((1, 256, 63, 63), dtype="float32") = split[0]
            lv4: R.Tensor((1, 256, 63, 63), dtype="float32") = split[1]
            out: R.Tensor((1, 256, 63, 63), dtype="float32") = R.add(lv3, lv4)
            R.output(out)
        return out
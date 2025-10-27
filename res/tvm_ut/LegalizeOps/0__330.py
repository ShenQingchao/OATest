# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(inp_0: R.Tensor((2, 320, 64, 64), dtype="float32"), inp_1: R.Tensor((2, 1280), dtype="float32"), w1: R.Tensor((320, 320, 3, 3), dtype="float32"), b1: R.Tensor((320,), dtype="float32"), w2: R.Tensor((320, 1280), dtype="float32"), b2: R.Tensor((320,), dtype="float32")) -> R.Tensor((2, 320, 64, 64), dtype="float32"):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv27: R.Tensor((2, 320, 64, 64), dtype="float32") = R.nn.conv2d(inp_0, w1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv28: R.Tensor((1, 320, 1, 1), dtype="float32") = R.reshape(b1, R.shape([1, 320, 1, 1]))
            lv29: R.Tensor((2, 320, 64, 64), dtype="float32") = R.add(lv27, lv28)
            lv31: R.Tensor((1280, 320), dtype="float32") = R.permute_dims(w2, axes=None)
            lv32: R.Tensor((2, 320), dtype="float32") = R.matmul(inp_1, lv31, out_dtype="float32")
            lv33: R.Tensor((2, 320), dtype="float32") = R.add(lv32, b2)
            lv35: R.Tensor((2, 320, 1, 1), dtype="float32") = R.reshape(lv33, R.shape([2, 320, 1, 1]))
            lv36: R.Tensor((2, 320, 64, 64), dtype="float32") = R.add(lv29, lv35)
            gv: R.Tensor((2, 320, 64, 64), dtype="float32") = lv36
            R.output(gv)
        return gv
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def unrelated_function(A: R.Tensor((16, 16), dtype="float16")) -> R.Tensor((16, 16), dtype="float16"):
        # from tvm.script import relax as R
        
        @R.function
        def inner_func(B: R.Tensor((16, 16), dtype="float16")) -> R.Tensor((16, 16), dtype="float16"):
            with R.dataflow():
                C: R.Tensor((16, 16), dtype="float16") = R.multiply(B, R.const(2, "float16"))
                R.output(C)
            return C

        D: R.Tensor((16, 16), dtype="float16") = inner_func(A)
        return D

    @R.function
    def main(data: R.Tensor((1, 64, 56, 56), dtype="float32"), weight1: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(data, weight1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            conv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv)
            R.output(conv1)
        return conv1
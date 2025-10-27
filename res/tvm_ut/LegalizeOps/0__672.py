# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((4, 112, 112, 6), dtype="float32")) -> R.Tensor((4, 56, 56, 6), dtype="float32"):
        gv: R.Tensor((4, 56, 56, 6), dtype="float32") = R.nn.max_pool2d(x, pool_size=[3, 3], strides=[2, 2], dilation=[1, 1], padding=[1, 1, 1, 1], ceil_mode=False, count_include_pad=False, layout="NHWC", out_layout="NHWC")
        return gv
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((4, 4, 112, 112, 16), dtype="float32")) -> R.Tensor((4, 4, 110, 110, 16), dtype="float32"):
        gv: R.Tensor((4, 4, 110, 110, 16), dtype="float32") = R.nn.avg_pool2d(x, pool_size=[3, 3], strides=[1, 1], dilation=[1, 1], padding=[0, 0, 0, 0], ceil_mode=False, count_include_pad=False, layout="NCHW16c", out_layout="NCHW16c")
        return gv
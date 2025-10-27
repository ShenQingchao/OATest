# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((4, 6, 112, 112), dtype="float32")) -> R.Tensor((4, 6, 38, 38), dtype="float32"):
        gv: R.Tensor((4, 6, 38, 38), dtype="float32") = R.nn.max_pool2d(x, pool_size=[3, 3], strides=[3, 3], dilation=[1, 1], padding=[1, 1, 1, 1], ceil_mode=True, count_include_pad=False, layout="NCHW", out_layout="NCHW")
        return gv
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 4, 7, 7, 16), dtype="float32")) -> R.Tensor((2, 4, 1, 1, 16), dtype="float32"):
        gv: R.Tensor((2, 4, 1, 1, 16), dtype="float32") = R.nn.adaptive_avg_pool2d(x, output_size=[1, 1], layout="NCHW16c", out_layout="NCHW16c")
        return gv
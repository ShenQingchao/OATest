# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 16, 7, 7), dtype="float32")) -> R.Tensor((2, 16, 7, 7), dtype="float32"):
        gv: R.Tensor((2, 16, 7, 7), dtype="float32") = R.nn.adaptive_avg_pool2d(x, output_size=None, layout="NCHW", out_layout="NCHW")
        return gv
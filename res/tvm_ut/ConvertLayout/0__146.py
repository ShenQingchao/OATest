# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(dtype="float32", ndim=4), w: R.Tensor(dtype="float32", ndim=4)) -> R.Tensor(dtype="float32", ndim=4):
        with R.dataflow():
            gv: R.Tensor(dtype="float32", ndim=4) = R.nn.conv2d(x, w, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            R.output(gv)
        return gv
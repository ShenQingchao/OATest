# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(dtype="float32", ndim=4), w: R.Tensor(dtype="float32", ndim=4)) -> R.Tensor(dtype="float32", ndim=4):
        with R.dataflow():
            lv: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(x, axes=[0, 2, 3, 1])
            lv1: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(w, axes=[0, 2, 3, 1])
            lv2: R.Tensor(dtype="float32", ndim=4) = R.nn.conv2d(lv, lv1, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="float32")
            gv: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(lv2, axes=[0, 3, 1, 2])
            R.output(gv)
        return gv
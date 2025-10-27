# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 52, 52), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(x, w, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            gv2: R.Tensor((2, 4, 52, 52), dtype="float32") = R.image.resize2d(gv, R.shape([52, 52]), roi=[T.float32(0), T.float32(0), T.float32(0), T.float32(0)], layout="NCHW", method="linear", coordinate_transformation_mode="half_pixel", rounding_method="round", cubic_alpha=-0.5, cubic_exclude=0, extrapolation_value=0, out_dtype="void")
            R.output(gv2)
        return gv2
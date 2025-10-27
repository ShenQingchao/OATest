# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 50, 50), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
            gv: R.Tensor((2, 52, 52, 3), dtype="float32") = R.image.resize2d(lv, R.shape([52, 52]), roi=[T.float32(0), T.float32(0), T.float32(0), T.float32(0)], layout="NHWC", method="linear", coordinate_transformation_mode="half_pixel", rounding_method="round", cubic_alpha=-0.5, cubic_exclude=0, extrapolation_value=0, out_dtype="void")
            lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
            lv2: R.Tensor((2, 50, 50, 4), dtype="float32") = R.nn.conv2d(gv, lv1, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="float32")
            gv2: R.Tensor((2, 4, 50, 50), dtype="float32") = R.permute_dims(lv2, axes=[0, 3, 1, 2])
            R.output(gv2)
        return gv2
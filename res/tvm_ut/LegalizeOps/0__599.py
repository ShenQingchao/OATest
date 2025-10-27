# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 8, 8, 3), dtype="float32")) -> R.Tensor((2, 16, 16, 3), dtype="float32"):
        gv: R.Tensor((2, 16, 16, 3), dtype="float32") = R.image.resize2d(x, R.shape([16, 16]), roi=[T.float32(0), T.float32(0), T.float32(0), T.float32(0)], layout="NHWC", method="nearest_neighbor", coordinate_transformation_mode="asymmetric", rounding_method="round", cubic_alpha=-0.5, cubic_exclude=0, extrapolation_value=0, out_dtype="void")
        return gv
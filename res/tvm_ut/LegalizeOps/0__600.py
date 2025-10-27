# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(dumb_param: R.Tensor(("oh", "ow")), x: R.Tensor(("n", "c", "h", "w", 16), dtype="float32")) -> R.Tensor(("n", "c", "oh", "ow", 16), dtype="float32"):
        n = T.int64()
        c = T.int64()
        oh = T.int64()
        ow = T.int64()
        h = T.int64()
        w = T.int64()
        gv: R.Tensor((n, c, oh, ow, 16), dtype="float32") = R.image.resize2d(x, R.shape([oh, ow]), roi=[T.float32(0), T.float32(0), T.float32(0), T.float32(0)], layout="NCHW16c", method="nearest_neighbor", coordinate_transformation_mode="asymmetric", rounding_method="round", cubic_alpha=-0.5, cubic_exclude=0, extrapolation_value=0, out_dtype="void")
        return gv
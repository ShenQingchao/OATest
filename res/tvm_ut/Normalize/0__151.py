# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(dtype="float32", ndim=4), w: R.Tensor(dtype="float32", ndim=4)) -> R.Tensor(dtype="float32", ndim=4):
        N = T.int64()
        H = T.int64()
        W = T.int64()
        C = T.int64()
        with R.dataflow():
            lv: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(x, axes=[0, 2, 3, 1])
            lv1: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(w, axes=[0, 2, 3, 1])
            lv0: R.Tensor(dtype="float32", ndim=4) = R.nn.conv2d(lv, lv1, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="float32")
            lv2: R.Tensor((N, H, W, C), dtype="float32") = R.match_cast(lv0, R.Tensor((N, H, W, C), dtype="float32"))
            lv3: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(w, axes=[0, 2, 3, 1])
            lv4: R.Tensor(dtype="float32", ndim=4) = R.add(lv2, lv3)
            gv: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(lv4, axes=[0, 3, 1, 2])
            R.output(gv)
        return gv
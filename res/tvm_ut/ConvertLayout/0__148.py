# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(dtype="float32", ndim=4), w: R.Tensor(dtype="float32", ndim=4)) -> R.Tensor(dtype="float32", ndim=4):
        N = T.int64()
        C = T.int64()
        H = T.int64()
        W = T.int64()
        with R.dataflow():
            lv0: R.Tensor((N, C, H, W), dtype="float32") = R.match_cast(x, R.Tensor((N, C, H, W), dtype="float32"))
            gv: R.Tensor(dtype="float32", ndim=4) = R.nn.conv2d(lv0, w, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            R.output(gv)
        return gv
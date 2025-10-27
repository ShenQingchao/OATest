# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def relu(data: T.Buffer((64, 64, 56, 56), "float32"), out: T.Buffer((64, 64, 56, 56), "float32")):
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(64, 64, 56, 56):
            with T.block("root"):
                i, j, k, l = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(data[i, j, k, l])
                T.writes(out[i, j, k, l])
                out[i, j, k, l] = T.max(data[i, j, k, l], T.float32(0))

    @R.function
    def main(data: R.Tensor((64, 64, 56, 56), dtype="float32"), weight1: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((64, 64, 56, 56), dtype="float32"):
        cls = Module
        with R.dataflow():
            conv1 = R.nn.conv2d(data, weight1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            relu1 = R.call_tir(cls.relu, (conv1,), out_sinfo=R.Tensor((64, 64, 56, 56), dtype="float32"))
            R.output(relu1)
        return relu1

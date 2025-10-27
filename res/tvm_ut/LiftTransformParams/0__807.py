# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def transform_layout_IOHW_to_OIHW(w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")):
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
            with T.block("layout_transform"):
                o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(w1[i, o, h, w])
                T.writes(out[o, i, h, w])
                out[o, i, h, w] = w1[i, o, h, w]

    @R.function
    def main(x: R.Tensor((1, 3, 224, 224), dtype="float32"), w1: R.Tensor((3, 16, 3, 3), dtype="float32"), w2: R.Tensor((16, 16, 3, 3), dtype="float32")) -> R.Tensor((1, 16, 224, 224), dtype="float32"):
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            w1_transformed = R.call_tir(cls.transform_layout_IOHW_to_OIHW, (w1,), out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"))
            conv1: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(x, w1_transformed, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            conv2: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(conv1, w2, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            R.output(conv2)
        return conv2
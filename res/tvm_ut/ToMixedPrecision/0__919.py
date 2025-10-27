# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 320), dtype="float32"), w1: R.Tensor((320, 1280), dtype="float32"), w2: R.Tensor((2, 1280), dtype="float32")) -> R.Tensor((2, 1280), dtype="float32"):
        with R.dataflow():
            gv0: R.Tensor((2, 1280), dtype="float32") = R.matmul(x, w1, out_dtype="float32")
            gv1: R.Tensor((2, 1280), dtype="float32") = R.add(gv0, w2)
            gv2: R.Tensor((2, 1280), dtype="float32") = R.nn.silu(gv1)
            R.output(gv2)
        return gv2
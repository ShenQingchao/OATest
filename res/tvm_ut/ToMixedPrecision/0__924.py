# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(lv10: R.Tensor((2, 160), dtype="float32"), lv12: R.Tensor((2, 160), dtype="float32"), w: R.Tensor((320, 1280), dtype="float32")) -> R.Tensor((2, 1280), dtype="float32"):
        with R.dataflow():
            lv13: R.Tensor((2, 320), dtype="float32") = R.concat((lv10, lv12), axis=-1)
            lv14: R.Tensor((2, 1280), dtype="float32") = R.matmul(lv13, w, out_dtype="float32")
            R.output(lv14)
        return lv14
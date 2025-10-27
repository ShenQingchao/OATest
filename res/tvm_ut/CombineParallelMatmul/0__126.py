# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((640, 640), dtype="float32"), y: R.Tensor((640, 640), dtype="float32")) -> R.Tensor((640, 640), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((640, 640), dtype="float32") = R.matmul(x, y, out_dtype="float32")
            lv1: R.Tensor((640, 640), dtype="float32") = R.concat((lv,), axis=1)
            R.output(lv1)
        return lv1
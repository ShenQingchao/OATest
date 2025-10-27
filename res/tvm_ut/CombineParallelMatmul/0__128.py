# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 1024, 640), dtype="float32"), y: R.Tensor((640, 640), dtype="float32"), y_1: R.Tensor((640, 640), dtype="float32"), y_2: R.Tensor((640, 640), dtype="float32")) -> R.Tensor((2, 3072, 640), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x, y, out_dtype="float32")
            lv1: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x, y_1, out_dtype="float32")
            lv2: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x, y_2, out_dtype="float32")
            lv3: R.Tensor((2, 3072, 640), dtype="float32") = R.concat((lv, lv1, lv2), axis=1)
            R.output(lv3)
        return lv3
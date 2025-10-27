# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1024, 640), dtype="float32"), w0: R.Tensor((2, 640, 640), dtype="float32"), w1: R.Tensor((3, 640, 640), dtype="float32"), w2: R.Tensor((2, 640, 640), dtype="float32"), w3: R.Tensor((2, 640, 640), dtype="float32")) -> R.Tuple(R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((3, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32")):
        with R.dataflow():
            lv0: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x, w0, out_dtype="void")
            lv1: R.Tensor((3, 1024, 640), dtype="float32") = R.matmul(x, w1, out_dtype="void")
            lv2: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x, w2, out_dtype="void")
            lv3: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x, w3, out_dtype="void")
            out: R.Tuple(R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((3, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32")) = lv0, lv1, lv2, lv3
            R.output(out)
        return out
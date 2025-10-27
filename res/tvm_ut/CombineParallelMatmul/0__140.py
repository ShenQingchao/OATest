# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def main(x1: R.Tensor((2, 1024, 640), dtype="float32"), x2: R.Tensor((2, 1024, 640), dtype="float32"), w0: R.Tensor((640, 640), dtype="float32"), w1: R.Tensor((640, 640), dtype="float32"), w2: R.Tensor((640, 640), dtype="float32"), w3: R.Tensor((640, 640), dtype="float32"), w4: R.Tensor((640, 640), dtype="float32")) -> R.Tuple(R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32")):
        with R.dataflow():
            lv0: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x1, w0, out_dtype="void")
            lv1: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x1, w1, out_dtype="void")
            lv2: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x1, w2, out_dtype="void")
            lv3: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x2, w3, out_dtype="void")
            lv4: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x2, w4, out_dtype="void")
            out: R.Tuple(R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32")) = lv0, lv1, lv2, lv3, lv4
            R.output(out)
        return out
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def main(x1: R.Tensor((2, 1024, 640), dtype="float32"), x2: R.Tensor((2, 1024, 640), dtype="float32"), w0: R.Tensor((640, 640), dtype="float32"), w1: R.Tensor((640, 640), dtype="float32"), w2: R.Tensor((640, 640), dtype="float32"), w3: R.Tensor((640, 640), dtype="float32"), w4: R.Tensor((640, 640), dtype="float32"), b0: R.Tensor((640,), dtype="float32"), b1: R.Tensor((640,), dtype="float32")) -> R.Tuple(R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32")):
        with R.dataflow():
            lv0: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x1, w0, out_dtype="void")
            lv3: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x2, w3, out_dtype="void")
            lv1: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x1, w1, out_dtype="void")
            lv5: R.Tensor((2, 1024, 640), dtype="float32") = R.add(lv3, b0)
            lv2: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x1, w2, out_dtype="void")
            lv4: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x2, w4, out_dtype="void")
            lv6: R.Tensor((2, 1024, 640), dtype="float32") = R.add(lv4, b1)
            out: R.Tuple(R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32")) = lv0, lv1, lv2, lv5, lv6
            R.output(out)
        return out
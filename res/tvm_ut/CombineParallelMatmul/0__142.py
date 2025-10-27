# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def main(x: R.Tensor((2, 1024, 640), dtype="float32"), w0: R.Tensor((640, "M"), dtype="float32"), w1: R.Tensor((640, 640), dtype="float32")) -> R.Tuple(R.Tensor((2, 1024, "M"), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32")):
        M = T.int64()
        with R.dataflow():
            lv0: R.Tensor((2, 1024, M), dtype="float32") = R.matmul(x, w0, out_dtype="void")
            lv1: R.Tensor((2, 1024, 640), dtype="float32") = R.matmul(x, w1, out_dtype="void")
            out: R.Tuple(R.Tensor((2, 1024, M), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32")) = lv0, lv1
            R.output(out)
        return out
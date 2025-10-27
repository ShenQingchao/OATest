# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 16), dtype="float16"), w: R.Tensor((16, 16), dtype="float16")) -> R.Tensor((1, 16), dtype="float16"):
        with R.dataflow():
            w1: R.Tensor((16, 16), dtype="float16") = R.permute_dims(w, axes=None)
            y: R.Tensor((1, 16), dtype="float16") = R.matmul(x, w1, out_dtype="void")
            y1: R.Tensor((1, 16), dtype="float16") = y
            out: R.Tensor((1, 16), dtype="float16") = R.add(x, y1)
            R.output(out)
        return out
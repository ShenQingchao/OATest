# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(c0: R.Tensor((16, 16), dtype="float32"), c1: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        with R.dataflow():
            lv0: R.Tensor((16, 16), dtype="float32") = R.add(c0, c1)
            lv1: R.Tensor((16, 16), dtype="float32") = R.multiply(c0, lv0)
            gv: R.Tensor((16, 16), dtype="float32") = R.subtract(lv1, c1)
            R.output(gv)
        return gv
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((3, 3), dtype="float32") = R.multiply(x, x)
            lv1: R.Tensor((3, 3), dtype="float32") = R.multiply(lv, y)
            lv2: R.Tensor((3, 3), dtype="float32") = R.multiply(lv1, y)
            gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
            R.output(gv)
        return gv
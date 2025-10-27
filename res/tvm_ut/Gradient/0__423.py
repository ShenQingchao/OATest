# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")):
        with R.dataflow():
            lv1: R.Tensor((3, 3), dtype="float32") = x
            lv2: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            lv3: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
            R.output(lv1, lv2, lv3)
        return (lv1, lv2, lv3)
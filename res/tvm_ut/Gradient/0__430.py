# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((6,), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv1: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.split(x, indices_or_sections=2, axis=0)
            lv2: R.Tensor((6,), dtype="float32") = R.concat(lv1, axis=0)
            gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
            R.output(gv)
        return gv
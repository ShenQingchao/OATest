# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
        with R.dataflow():
            lv1: R.Tensor((1,), dtype="float32") = R.reshape(x, R.shape([1]))
            lv2: R.Tensor((1,), dtype="float32") = R.add(lv1, lv1)
            R.output(lv2)
        return lv2
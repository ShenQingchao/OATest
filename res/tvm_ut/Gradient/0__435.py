# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            s: R.Shape([3, 2, 2]) = R.shape([3, 2, 2])
            lv: R.Tensor((3, 2, 2), dtype="float32") = R.reshape(x, s)
            gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
            R.output(gv)
        return gv
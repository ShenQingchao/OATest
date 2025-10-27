# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32"))):
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            R.output(gv)
        return (gv, (gv, gv))
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x0: R.Tensor(("n", "n"), dtype="float32"), x1: R.Tensor(("n", "n"), dtype="float32"), x2: R.Tensor(("n", "n"), dtype="float32"), x3: R.Tensor(("n", "n"), dtype="float32")) -> R.Tensor((), dtype="float32"):
        n = T.int64()
        with R.dataflow():
            lv0: R.Tensor((n, n), dtype="float32") = R.add(x0, x1)
            lv1: R.Tensor((n, n), dtype="float32") = R.add(x2, x3)
            lv2: R.Tensor((n, n), dtype="float32") = R.add(lv0, lv1)
            gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
            R.output(gv)
        return gv
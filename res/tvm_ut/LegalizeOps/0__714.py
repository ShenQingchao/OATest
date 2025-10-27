# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(predictions: R.Tensor(("N", "C", "d1", "d2"), dtype="float32"), targets: R.Tensor(("N", "d1", "d2"), dtype="int64"), weights: R.Tensor(("C",), dtype="float32")) -> R.Tensor((), dtype="float32"):
        N = T.int64()
        C = T.int64()
        d1 = T.int64()
        d2 = T.int64()
        gv: R.Tensor((), dtype="float32") = R.nn.nll_loss(predictions, targets, weights, reduction="mean", ignore_index=-1)
        return gv
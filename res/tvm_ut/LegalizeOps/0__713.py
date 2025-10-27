# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(predictions: R.Tensor(("C",), dtype="float32"), targets: R.Tensor((), dtype="int64"), weights: R.Tensor(("C",), dtype="float32")) -> R.Tensor((), dtype="float32"):
        C = T.int64()
        gv: R.Tensor((), dtype="float32") = R.nn.nll_loss(predictions, targets, weights, reduction="mean", ignore_index=1)
        return gv
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(predictions: R.Tensor((2, 3, 4, 5), dtype="float32"), targets: R.Tensor((2, 4, 5), dtype="int64"), weights: R.Tensor((4,), dtype="float32")) -> R.Tensor((), dtype="float32"):
        gv: R.Tensor((), dtype="float32") = R.nn.nll_loss(predictions, targets, weights, reduction="mean", ignore_index=-1)
        return gv
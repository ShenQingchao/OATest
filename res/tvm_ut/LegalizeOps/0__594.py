# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(output_grad: R.Tensor((), dtype="float32"), predictions: R.Tensor((4,), dtype="float32"), targets: R.Tensor((), dtype="int64"), weights: R.Tensor((4,), dtype="float32")) -> R.Tensor((4,), dtype="float32"):
        gv: R.Tensor((4,), dtype="float32") = R.grad.nll_loss_backward(output_grad, predictions, targets, weights, reduction="mean", ignore_index=-1)
        return gv
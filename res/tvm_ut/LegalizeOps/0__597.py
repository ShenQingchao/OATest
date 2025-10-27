# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(output_grad: R.Tensor((3, 2, 5), dtype="float32"), x: R.Tensor((3, 4, 5), dtype="float32"), indices: R.Tensor((2,), dtype="int32")) -> R.Tensor((3, 4, 5), dtype="float32"):
        gv: R.Tensor((3, 4, 5), dtype="float32") = R.grad.take_backward(output_grad, x, indices, axis=1)
        return gv
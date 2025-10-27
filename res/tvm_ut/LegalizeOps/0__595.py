# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(output_grad: R.Tensor((3, 2, 6, 5), dtype="float32"), data: R.Tensor((3, 2, 10, 10), dtype="float32")) -> R.Tensor((3, 2, 10, 10), dtype="float32"):
        gv: R.Tensor((3, 2, 10, 10), dtype="float32") = R.grad.max_pool2d_backward(output_grad, data, pool_size=[5, 5], strides=[2, 2], dilation=[1, 1], padding=[2, 1, 2, 1], ceil_mode=True, count_include_pad=False, layout="NCHW", out_layout="NCHW")
        return gv
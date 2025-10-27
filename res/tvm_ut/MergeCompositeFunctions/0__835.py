# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def fused_relax_matmul(lv: R.Tensor((1, 784), dtype="float32"), lv1: R.Tensor((784, 512), dtype="float32")) -> R.Tensor((1, 512), dtype="float32"):
        R.func_attr({"Composite": "tensorrt.matmul", "Primitive": 1})
        with R.dataflow():
            gv: R.Tensor((1, 512), dtype="float32") = R.matmul(lv, lv1, out_dtype="float32")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_reshape(inp_0: R.Tensor((1, 1, 28, 28), dtype="float32"), param_0: R.Shape([1, 784])) -> R.Tensor((1, 784), dtype="float32"):
        R.func_attr({"Composite": "tensorrt.reshape", "Primitive": 1})
        with R.dataflow():
            gv: R.Tensor((1, 784), dtype="float32") = R.reshape(inp_0, param_0)
            R.output(gv)
        return gv

    @R.function
    def main(inp_0: R.Tensor((1, 1, 28, 28), dtype="float32"), linear_relu_stack_0_weight: R.Tensor((512, 784), dtype="float32")) -> R.Tensor((1, 512), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((1, 784), dtype="float32") = cls.fused_relax_reshape(inp_0, R.shape([1, 784]))
            lv1: R.Tensor((784, 512), dtype="float32") = R.permute_dims(linear_relu_stack_0_weight, axes=None)
            lv_1: R.Tensor((1, 512), dtype="float32") = cls.fused_relax_matmul(lv, lv1)
            gv: R.Tensor((1, 512), dtype="float32") = lv_1
            R.output(gv)
        return gv
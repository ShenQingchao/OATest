# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(inp_0: R.Tensor((1, 784), dtype="float32"), inp_1: R.Tensor((1, 128), dtype="float32"), linear1_bias: R.Tensor((128,), dtype="float32"), linear1_weight: R.Tensor((128, 784), dtype="float32"), linear2_bias: R.Tensor((10,), dtype="float32"), linear2_weight: R.Tensor((10, 128), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(linear1_weight, axes=None)
            lv1: R.Tensor((1, 128), dtype="float32") = R.matmul(inp_0, lv, out_dtype="float32")
            lv2: R.Tensor((1, 128), dtype="float32") = R.add(lv1, linear1_bias)
            lv3: R.Tensor((1, 128), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(linear2_weight, axes=None)
            lv5: R.Tensor((1, 10), dtype="float32") = R.matmul(inp_1, lv4, out_dtype="float32")
            lv6: R.Tensor((1, 10), dtype="float32") = R.add(lv5, linear2_bias)
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return gv
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def func1(x: R.Tensor((256, 256), dtype="float32"), w1: R.Tensor((256, 256), dtype="float32"), w2: R.Tensor((256, 256), dtype="float32")) -> R.Tensor((256, 256), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            w1_t: R.Tensor((256, 256), dtype="float32") = R.permute_dims(w1, axes=None)
            y1: R.Tensor((256, 256), dtype="float32") = R.matmul(x, w1_t, out_dtype="void")
            w2_t: R.Tensor((256, 256), dtype="float32") = R.permute_dims(w2, axes=None)
            y2: R.Tensor((256, 256), dtype="float32") = R.matmul(x, w2_t, out_dtype="void")
            output: R.Tensor((256, 256), dtype="float32") = R.add(y1, y2)
            R.output(output)
        return output

    @R.function
    def func2(x: R.Tensor((256, 256), dtype="float32"), w1: R.Tensor((128, 256), dtype="float32"), w2: R.Tensor((128, 256), dtype="float32")) -> R.Tensor((256, 128), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            w1_t: R.Tensor((256, 128), dtype="float32") = R.permute_dims(w1, axes=None)
            y1: R.Tensor((256, 128), dtype="float32") = R.matmul(x, w1_t, out_dtype="void")
            w2_t: R.Tensor((256, 128), dtype="float32") = R.permute_dims(w2, axes=None)
            y2: R.Tensor((256, 128), dtype="float32") = R.matmul(x, w2_t, out_dtype="void")
            output: R.Tensor((256, 128), dtype="float32") = R.multiply(y1, y2)
            R.output(output)
        return output
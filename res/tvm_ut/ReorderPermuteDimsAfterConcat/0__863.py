# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 32), dtype="float32"), weight_A: R.Tensor((32, 128), dtype="float32"), linear_weight_B: R.Tensor((128, 32), dtype="float32")) -> R.Tensor((1, 256), dtype="float32"):
        with R.dataflow():
            matmul_weight_A: R.Tensor((32, 128), dtype="float32") = R.permute_dims(weight_A, axes=[0, 1])
            matmul_weight_B: R.Tensor((32, 128), dtype="float32") = R.permute_dims(linear_weight_B, axes=[1, 0])
            matmul_weight: R.Tensor((32, 256), dtype="float32") = R.concat((matmul_weight_A, matmul_weight_B), axis=1)
            out: R.Tensor((1, 256), dtype="float32") = R.matmul(x, matmul_weight, out_dtype="void")
            R.output(out)
        return out
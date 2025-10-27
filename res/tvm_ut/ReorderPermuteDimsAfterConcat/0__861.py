# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 32), dtype="float32"), weight_A: R.Tensor((32, 128), dtype="float32"), linear_weight_B: R.Tensor((128, 32), dtype="float32"), linear_weight_C: R.Tensor((128, 32), dtype="float32"), linear_weight_D: R.Tensor((128, 32), dtype="float32")) -> R.Tuple(R.Tensor((1, 256), dtype="float32"), R.Tensor((1, 256), dtype="float32")):
        with R.dataflow():
            matmul_weight_C: R.Tensor((32, 128), dtype="float32") = R.permute_dims(linear_weight_C, axes=None)
            matmul_weight_D: R.Tensor((32, 128), dtype="float32") = R.permute_dims(linear_weight_D, axes=None)
            matmul_weight_CD: R.Tensor((32, 256), dtype="float32") = R.concat((matmul_weight_C, matmul_weight_D), axis=1)
            out_CD: R.Tensor((1, 256), dtype="float32") = R.matmul(x, matmul_weight_CD, out_dtype="void")
            matmul_weight_A: R.Tensor((32, 128), dtype="float32") = R.permute_dims(weight_A, axes=[0, 1])
            matmul_weight_B: R.Tensor((32, 128), dtype="float32") = R.permute_dims(linear_weight_B, axes=[1, 0])
            matmul_weight_AB: R.Tensor((32, 256), dtype="float32") = R.concat((matmul_weight_A, matmul_weight_B), axis=1)
            out_AB: R.Tensor((1, 256), dtype="float32") = R.matmul(x, matmul_weight_AB, out_dtype="void")
            out: R.Tuple(R.Tensor((1, 256), dtype="float32"), R.Tensor((1, 256), dtype="float32")) = out_AB, out_CD
            R.output(out)
        return out
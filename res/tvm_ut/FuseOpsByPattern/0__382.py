# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("batch_size", 1024), dtype="float16"), w1: R.Tensor((1024, 1024), dtype="float16"), w2: R.Tensor((1024, "M"), dtype="float16")) -> R.Tuple(R.Tensor(("batch_size", 1024), dtype="float16"), R.Tensor(("batch_size", "M"), dtype="float16")):
        batch_size = T.int64()
        M = T.int64()
        with R.dataflow():
            matmul1: R.Tensor((batch_size, 1024), dtype="float16") = R.matmul(x, w1, out_dtype="void")
            matmul2: R.Tensor((batch_size, M), dtype="float16") = R.matmul(x, w2, out_dtype="void")
            out: R.Tuple(R.Tensor((batch_size, 1024), dtype="float16"), R.Tensor((batch_size, M), dtype="float16")) = matmul1, matmul2
            R.output(out)
        return out
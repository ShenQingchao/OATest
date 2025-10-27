# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def transform_params(A: R.Tensor(("vocab_size", 4096), dtype="float16"), B: R.Tensor((6144, 4096), dtype="float16")) -> R.Tuple(R.Tensor(("vocab_size", 4096), dtype="float16"), R.Tensor((6144, 4096), dtype="float16")):
        vocab_size = T.int64()
        with R.dataflow():
            C: R.Tensor((6144, 4096), dtype="float16") = B
            D: R.Tuple(R.Tensor((vocab_size, 4096), dtype="float16"), R.Tensor((6144, 4096), dtype="float16")) = A, C
            E: R.Tuple(R.Tensor((vocab_size, 4096), dtype="float16"), R.Tensor((6144, 4096), dtype="float16")) = D
            R.output(E)
        return E
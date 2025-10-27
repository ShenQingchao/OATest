# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((10, "n"), dtype="float32"), begin: R.Tensor((2,), dtype="int64"), end: R.Tensor((2,), dtype="int64"), strides: R.Tensor((2,), dtype="int64")) -> R.Tensor(dtype="float32", ndim=2):
        n = T.int64()
        gv: R.Tensor(dtype="float32", ndim=2) = R.dynamic_strided_slice(x, begin, end, strides)
        return gv
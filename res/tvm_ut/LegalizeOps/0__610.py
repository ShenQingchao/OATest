# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((8, 9, 10, 10), dtype="float32"), begin: R.Tensor((4,), dtype="int64"), end: R.Tensor((4,), dtype="int64"), strides: R.Tensor((4,), dtype="int64")) -> R.Tensor(dtype="float32", ndim=4):
        gv: R.Tensor(dtype="float32", ndim=4) = R.dynamic_strided_slice(x, begin, end, strides)
        return gv
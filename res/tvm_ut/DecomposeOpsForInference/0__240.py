# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(t: R.Tensor(dtype="int64", ndim=1)) -> R.Shape(ndim=3):
        gv: R.Shape(ndim=3) = R.tensor_to_shape(t)
        return gv
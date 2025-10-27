# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(v: R.Tensor((), dtype="int32")) -> R.Tensor((2, 3), dtype="int32"):
        gv: R.Tensor((2, 3), dtype="int32") = R.full(R.shape([2, 3]), v, dtype="int32")
        return gv
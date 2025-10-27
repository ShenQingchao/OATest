# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(t: R.Tuple(R.Tensor((3, 4), dtype="float32"), R.Tensor((3, 5), dtype="float32"))) -> R.Tensor((3, 9), dtype="float32"):
        gv: R.Tensor((3, 9), dtype="float32") = R.concat(t, axis=1)
        return gv
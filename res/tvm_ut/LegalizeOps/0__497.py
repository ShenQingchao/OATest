# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 3), dtype="bool")) -> R.Tensor((3, 3), dtype="bool"):
        gv: R.Tensor((3, 3), dtype="bool") = R.equal(x, R.const(True, "bool"))
        return gv
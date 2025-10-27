# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def f(x: R.Tensor(dtype="float32")) -> R.Tuple(R.Tensor(dtype="float32"), R.Tensor(dtype="float32")):
        gv: R.Tensor(dtype="float32") = R.add(x, x)
        gv1: R.Tensor(dtype="float32") = R.add(gv, gv)
        gv2: R.Tensor(dtype="float32") = R.add(gv, gv1)
        return (gv, gv2)
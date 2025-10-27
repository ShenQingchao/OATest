# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def main(cond: R.Tensor((), dtype="bool"), x: R.Tensor((), dtype="int32"), y: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        if cond:
            z = R.add(x, y)
        else:
            z = R.multiply(x, y)
        return z
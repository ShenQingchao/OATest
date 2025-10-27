# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def main(cond: R.Tensor((), dtype="bool"), x: R.Tensor((1,), dtype="float32")) -> R.Tensor(dtype="float32", ndim=1):
        if cond:
            y = R.multiply(R.add(x, x), R.add(x, x))
        else:
            y = R.add(R.multiply(x, x), R.multiply(x, x))
        return y
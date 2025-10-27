# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        z: R.Tensor = R.add(x, y)
        w: R.Tensor = R.multiply(z, y)
        v: R.Tensor = R.add(w, x)
        R.print(format=R.str("Hi mom!"))
        a: R.Tensor = R.multiply(v, v)
        b: R.Tensor = R.add(a, a)
        c: R.Tensor = R.subtract(b, a)
        d: R.Tensor = R.add(c, c)
        return d
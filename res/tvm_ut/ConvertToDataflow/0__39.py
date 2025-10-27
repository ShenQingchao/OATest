# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def func(A: R.Tensor, B: R.Tensor) -> R.Tensor:
        x: R.Tensor = R.add(A, B)
        return x

    @R.function(pure=False)
    def func2(A: R.Tensor, B: R.Tensor) -> R.Tensor:
        R.print(format=R.str("Hi there!"))
        y: R.Tensor = R.add(A, B)
        R.print(y, format=R.str("Sum: {}"))
        x: R.Tensor = R.multiply(y, y)
        if R.const(False, "bool"):
            R.print(format=R.str("True branch"))
            q: R.Tensor = R.add(x, y)
            R.print(q, format=R.str("Value of q: {}"))
            w: R.Tensor = q
        else:
            R.print(format=R.str("False branch"))
            q: R.Tensor = R.subtract(x, y)
            R.print(q, format=R.str("Value of q: {}"))
            w: R.Tensor = q
        p: R.Tensor = R.multiply(w, w)
        return p

    @R.function
    def main(A: R.Tensor, B: R.Tensor) -> R.Tensor:
        with R.dataflow():
            x: R.Tensor = R.add(A, B)
            y: R.Tensor = R.multiply(x, A)
            z: R.Tensor = R.add(x, y)
            q: R.Tensor = R.multiply(y, z)
            p: R.Tensor = R.add(z, q)
            R.output(p)
        return p
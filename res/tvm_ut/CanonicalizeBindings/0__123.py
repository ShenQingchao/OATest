# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        with R.dataflow():
            # from tvm.script import relax as R
            
            @R.function(pure=False)
            def inner_func(x_1: R.Tensor, y_1: R.Tensor) -> R.Tensor:
                with R.dataflow():
                    z: R.Tensor = R.add(x_1, y_1)
                    w: R.Tensor = R.multiply(x_1, z)
                    v: R.Tensor = R.add(y_1, w)
                    R.output(z, w, v)
                R.print(format=R.str("oops"))
                with R.dataflow():
                    a: R.Tensor = R.multiply(v, v)
                    b: R.Tensor = R.add(a, a)
                    c: R.Tensor = R.multiply(a, b)
                    R.output(a, b, c)
                return c

            z: R.Tensor = R.add(x, y)
            w: R.Tensor = R.multiply(z, z)
            v: R.Tensor = R.divide(w, z)
            R.output(inner_func, z, w, v)
        q: R.Tensor = inner_func(w, v)
        with R.dataflow():
            a: R.Tensor = R.multiply(q, q)
            b: R.Tensor = R.add(a, a)
            c: R.Tensor = R.multiply(b, a)
            R.output(a, b, c)
        return c
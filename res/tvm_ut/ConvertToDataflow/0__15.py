# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        z: R.Tensor = R.add(x, y)
        w: R.Tensor = R.multiply(z, y)
        v: R.Tensor = R.add(w, x)
        if R.const(True, "bool"):
            q: R.Tensor = R.multiply(v, v)
            a: R.Tensor = R.add(q, q)
            b_then: R.Tensor = R.multiply(a, a)
            b: R.Tensor = b_then
        else:
            q: R.Tensor = R.add(v, v)
            a: R.Tensor = R.multiply(q, q)
            b_else: R.Tensor = R.add(a, a)
            b: R.Tensor = b_else
        c: R.Tensor = R.multiply(b, b)
        d: R.Tensor = R.add(c, c)
        e: R.Tensor = R.multiply(d, d)
        return e
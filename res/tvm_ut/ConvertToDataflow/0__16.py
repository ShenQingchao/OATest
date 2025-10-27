# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def extra(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        z: R.Tensor = R.add(x, y)
        q: R.Tensor = R.matmul(z, x, out_dtype="void")
        w: R.Tensor = R.nn.relu(q)
        R.print(format=R.str("Whoa"))
        return w

    @R.function(pure=False)
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        cls = Module
        z: R.Tensor = R.add(x, y)
        w: R.Tensor = R.multiply(z, z)
        q: R.Tensor = cls.extra(z, w)
        return q
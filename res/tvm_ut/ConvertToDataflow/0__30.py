# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        # from tvm.script import relax as R
        
        @R.function
        def inner_func(x_1: R.Tensor, y_1: R.Tensor) -> R.Tensor:
            z: R.Tensor = R.add(x_1, y_1)
            w: R.Tensor = R.multiply(x_1, z)
            v: R.Tensor = R.add(y_1, w)
            return v

        z: R.Tensor = R.add(x, y)
        w: R.Tensor = R.multiply(z, z)
        v: R.Tensor = R.divide(w, z)
        q: R.Tensor = inner_func(w, v)
        return q
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        # from tvm.script import relax as R
        
        @R.function
        def outer_func(c1: R.Tensor((2, 3), dtype="float32")) -> R.Callable((R.Tensor((2, 3), dtype="float32"),), R.Tensor((2, 3), dtype="float32"), True):
            # from tvm.script import relax as R
            
            @R.function
            def inner_func(x1: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
                s: R.Tensor((2, 3), dtype="float32") = R.add(x1, c1)
                return s

            return inner_func

        in_call: R.Callable((R.Tensor((2, 3), dtype="float32"),), R.Tensor((2, 3), dtype="float32"), True) = outer_func(x)
        res: R.Tensor((2, 3), dtype="float32") = in_call(y)
        return res
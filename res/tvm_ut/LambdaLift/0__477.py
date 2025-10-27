# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def main(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        # from tvm.script import relax as R
        
        @R.function(pure=False)
        def inner() -> R.Tuple:
            R.print(format=R.str("Wow!"))
            return R.tuple()

        inner()
        return x
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main_inner() -> R.Tuple:
        return R.tuple()

    @R.function
    def main(x1: R.Tensor((10, 5), dtype="float32"), y1: R.Tensor((10, 5), dtype="float32")) -> R.Tensor((10, 5), dtype="float32"):
        # from tvm.script import relax as R
        
        @R.function
        def inner(x2: R.Tensor((10, 5), dtype="float32"), y2: R.Tensor((10, 5), dtype="float32")) -> R.Tensor((10, 5), dtype="float32"):
            s: R.Tensor((10, 5), dtype="float32") = R.add(x2, y2)
            return s

        gv1: R.Tensor((10, 5), dtype="float32") = inner(x1, y1)
        return gv1
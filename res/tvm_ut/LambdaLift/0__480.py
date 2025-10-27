# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x1: R.Tensor(("n", "m"), dtype="float32"), y1: R.Tensor(("n", "m"), dtype="float32")) -> R.Tensor(("n", "m"), dtype="float32"):
        n = T.int64()
        m = T.int64()
        # from tvm.script import tir as T
        # from tvm.script import relax as R
        
        @R.function
        def inner(x2: R.Tensor((n, m), dtype="float32"), y2: R.Tensor((n, m), dtype="float32")) -> R.Tensor((n, m), dtype="float32"):
            sum_inner: R.Tensor((n, m), dtype="float32") = R.add(x2, y2)
            return sum_inner

        sum_main: R.Tensor((n, m), dtype="float32") = inner(x1, y1)
        return sum_main
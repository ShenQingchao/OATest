# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def main(x: R.Tensor(("m", "n"), dtype="float16")) -> R.Tensor(dtype="float16", ndim=2):
        m = T.int64()
        n = T.int64()
        return R.multiply(R.add(x, x), R.add(x, x))
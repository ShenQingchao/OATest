# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(dumb_param: R.Tensor(("n",)), x: R.Tensor(("m", "n * 3"), dtype="float32")) -> R.Tuple(R.Tensor(("m", "n"), dtype="float32"), R.Tensor(("m", "n"), dtype="float32"), R.Tensor(("m", "n"), dtype="float32")):
        m = T.int64()
        n = T.int64()
        gv: R.Tuple(R.Tensor((m, n), dtype="float32"), R.Tensor((m, n), dtype="float32"), R.Tensor((m, n), dtype="float32")) = R.split(x, indices_or_sections=3, axis=1)
        return gv
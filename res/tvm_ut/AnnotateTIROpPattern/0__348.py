# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(A: R.Tensor((10, 20), dtype="float32")) -> R.Tensor(dtype="float32", ndim=2):
        m = T.int64()
        n = T.int64()
        with R.dataflow():
            lv: R.Tensor((m, n), dtype="float32") = R.match_cast(A, R.Tensor((m, n), dtype="float32"))
            gv: R.Tensor((m, n), dtype="float32") = lv
            R.output(gv)
        return gv
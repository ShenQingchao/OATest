# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("b", 1, "m", "k"), dtype="float32"), y: R.Tensor(("a", 1, "c", "k", "n"), dtype="float32")) -> R.Tensor(("a", "b", "c", "m", "n"), dtype="float32"):
        a = T.int64()
        b = T.int64()
        c = T.int64()
        m = T.int64()
        n = T.int64()
        k = T.int64()
        gv: R.Tensor((a, b, c, m, n), dtype="float32") = R.matmul(x, y, out_dtype="void")
        return gv
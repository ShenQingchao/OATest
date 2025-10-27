# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("a", "b"), dtype="float32"), indices: R.Tensor(("m", "n"), dtype="int64"), updates: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("a", "b"), dtype="float32"):
        a = T.int64()
        b = T.int64()
        m = T.int64()
        n = T.int64()
        gv: R.Tensor((a, b), dtype="float32") = R.scatter_elements(x, indices, updates, axis=1, reduction="update")
        return gv
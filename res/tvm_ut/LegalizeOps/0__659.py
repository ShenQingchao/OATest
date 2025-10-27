# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor(("a", "b", "c"), dtype="float32")) -> R.Tensor(("a", "c", "(b - b % -3) // 3", 3), dtype="float32"):
        a = T.int64()
        c = T.int64()
        b = T.int64()
        gv: R.Tensor((a, c, (b - b % -3) // 3, 3), dtype="float32") = R.layout_transform(x, index_map=T.index_map(lambda a_1, b_1, c_1: (a_1, c_1, b_1 // 3, b_1 % 3)), pad_value=R.prim_value(T.float32(2)), axis_separators=[])
        return gv
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((10, 21, 30), dtype="float32")) -> R.Tensor((10, 30, 7, 3), dtype="float32"):
        gv: R.Tensor((10, 30, 7, 3), dtype="float32") = R.layout_transform(x, index_map=T.index_map(lambda a, b, c: (a, c, b // 3, b % 3)), pad_value=R.prim_value(T.float32(2)), axis_separators=[])
        return gv
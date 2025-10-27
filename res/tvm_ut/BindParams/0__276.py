# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((5, 4, 3, 2), dtype="float32"), indices: R.Tensor((1,), dtype="int64")) -> R.Tensor((1, 1), dtype="int64"):
        with R.dataflow():
            lv: R.Tensor((4,), dtype="int64") = R.shape_to_tensor(R.shape([5, 4, 3, 2]))
            lv1: R.Tensor((1,), dtype="int64") = R.take(lv, indices, axis=0)
            lv2: R.Tensor((1, 1), dtype="int64") = R.expand_dims(lv1, axis=[0])
            gv: R.Tensor((1, 1), dtype="int64") = R.concat((lv2,), axis=0)
            R.output(gv)
        return gv
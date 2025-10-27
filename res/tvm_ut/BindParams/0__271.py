# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((256,), dtype="float32"), c0: R.Tensor((2,), dtype="int64"), c1: R.Tensor((2,), dtype="int64")) -> R.Tensor(dtype="float32", ndim=2):
        with R.dataflow():
            lv0: R.Tensor((2,), dtype="int64") = R.add(c0, c0)
            target_shape: R.Tensor((2,), dtype="int64") = R.multiply(lv0, c1)
            lv2: R.Shape(ndim=2) = R.tensor_to_shape(target_shape)
            gv: R.Tensor(dtype="float32", ndim=2) = R.reshape(data, lv2)
            R.output(gv)
        return gv
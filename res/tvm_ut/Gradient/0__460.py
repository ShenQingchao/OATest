# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 5), dtype="float32"), y: R.Tensor((5,), dtype="float32"), z: R.Tensor((5,), dtype="float32"), u: R.Tensor((5,), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv1: R.Tensor((3, 5), dtype="float32") = R.add(x, x)
            lv2: R.Tensor((3, 5), dtype="float32") = R.subtract(lv1, y)
            lv3: R.Tensor((3, 5), dtype="float32") = R.subtract(lv2, y)
            lv4: R.Tensor((5,), dtype="float32") = R.add(y, z)
            lv5: R.Tensor((3, 5), dtype="float32") = R.multiply(lv3, lv4)
            lv6: R.Tensor((), dtype="float32") = R.sum(lv5, axis=None, keepdims=False)
            R.output(lv6)
        return lv6
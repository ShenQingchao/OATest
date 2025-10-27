# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3,), dtype="float32"), y: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32"))) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv1: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = x, x
            lv2: R.Tensor((6,), dtype="float32") = R.concat(lv1, axis=0)
            lv3: R.Tensor((6,), dtype="float32") = R.concat((x, x), axis=0)
            lv4: R.Tensor((6,), dtype="float32") = R.concat(y, axis=0)
            lv5: R.Tensor((6,), dtype="float32") = R.add(lv2, lv3)
            lv6: R.Tensor((6,), dtype="float32") = R.add(lv5, lv4)
            gv: R.Tensor((), dtype="float32") = R.sum(lv6, axis=None, keepdims=False)
            R.output(gv)
        return gv
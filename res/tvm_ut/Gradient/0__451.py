# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tuple(R.Tensor((3, 3), dtype="int64"), R.Tensor((3, 3), dtype="int64"))) -> R.Tensor((), dtype="int64"):
        with R.dataflow():
            lv1: R.Tensor((3, 3), dtype="int64") = x[0]
            lv2: R.Tensor((3, 3), dtype="int64") = x[1]
            lv3: R.Tensor((3, 3), dtype="int64") = R.add(lv1, lv2)
            gv: R.Tensor((), dtype="int64") = R.sum(lv3, axis=None, keepdims=False)
            R.output(gv)
        return gv
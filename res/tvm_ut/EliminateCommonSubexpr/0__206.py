# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        with R.dataflow():
            tup: R.Tuple(R.Tuple(R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")), R.Tuple(R.Tensor((), dtype="int32"), R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")))), R.Tuple(R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")), R.Tuple(R.Tensor((), dtype="int32"), R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")))), R.Tuple(R.Tensor((), dtype="int32"), R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")))) = ((x, x), (x, (x, x))), ((x, x), (x, (x, x))), (x, (x, x))
            lv1: R.Tuple(R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")), R.Tuple(R.Tensor((), dtype="int32"), R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")))) = tup[0]
            lv2: R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")) = lv1[0]
            gv: R.Tensor((), dtype="int32") = lv2[1]
            R.output(gv)
        return gv
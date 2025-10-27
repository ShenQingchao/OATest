# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main() -> R.Tensor((1,), dtype="int64"):
        with R.dataflow():
            gv: R.Tensor((1,), dtype="int64") = R.expand_dims(R.const(0, "int64"), axis=[0])
            R.output(gv)
        return gv
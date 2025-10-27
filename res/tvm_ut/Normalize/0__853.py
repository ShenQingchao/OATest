# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def main(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        with R.dataflow():
            v0: R.Tensor((), dtype="int32") = x
            R.output(v0)
        with R.dataflow():
            v1: R.Tensor((), dtype="int32") = v0
            R.output(v1)
        v2: R.Tensor((), dtype="int32") = v1
        v3: R.Tensor((), dtype="int32") = v2
        return v3
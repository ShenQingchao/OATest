# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def main(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        idx_var: R.Tuple(R.Tensor((), dtype="int32")) = ((x,),)[0]
        ret: R.Tensor((), dtype="int32") = idx_var[0]
        return ret
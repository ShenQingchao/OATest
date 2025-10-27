# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def subroutine() -> R.Tensor((), dtype="int64"):
        R.func_attr({"relax.force_pure": 1})
        cls = Module
        cond: R.Tensor((), dtype="bool") = R.call_packed("dummy_function", sinfo_args=(R.Tensor((), dtype="bool"),))
        if cond:
            Out_then: R.Tensor((), dtype="int64") = cls.subroutine()
            Out: R.Tensor((), dtype="int64") = Out_then
        else:
            Out: R.Tensor((), dtype="int64") = R.const(0, "int64")
        return Out

    @R.function
    def main() -> R.Tensor((), dtype="int64"):
        cls = Module
        B: R.Tensor((), dtype="int64") = cls.subroutine()
        return B
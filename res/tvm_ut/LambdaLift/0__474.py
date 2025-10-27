# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        # from tvm.script import relax as R
        
        @R.function
        def while_loop(i: R.Tensor((), dtype="int32"), s: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            cond: R.Tensor((), dtype="bool") = R.call_pure_packed("test.vm.less", i, R.const(10, "int32"), sinfo_args=(R.Tensor((), dtype="bool"),))
            c: R.Tensor((), dtype="int32") = R.const(1, "int32")
            if cond:
                new_i: R.Tensor((), dtype="int32") = R.add(i, c)
                new_s: R.Tensor((2, 3), dtype="float32") = R.add(s, x)
                r_then: R.Tensor((2, 3), dtype="float32") = while_loop(new_i, new_s)
                r: R.Tensor((2, 3), dtype="float32") = r_then
            else:
                r: R.Tensor((2, 3), dtype="float32") = s
            return r

        gv: R.Tensor((2, 3), dtype="float32") = while_loop(R.const(0, "int32"), x)
        return gv
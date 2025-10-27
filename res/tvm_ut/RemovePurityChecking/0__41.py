# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def base(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        y: R.Tensor((), dtype="int32") = R.add(x, x)
        z: R.Tensor((), dtype="int32") = R.add(x, y)
        return z

    @R.function(pure=False)
    def impure_func() -> R.Tuple:
        R.print(format=R.str("I am impure!"))
        return R.tuple()

    @R.function(pure=False)
    def nested_impure_func() -> R.Tensor((), dtype="int32"):
        # from tvm.script import relax as R
        
        @R.function(pure=False)
        def nested() -> R.Tuple:
            R.print(format=R.str("Oops!"))
            return R.tuple()

        y: R.Tensor((), dtype="int32") = R.const(1, "int32")
        nested()
        return y

    @R.function
    def nested_pure_func() -> R.Tensor((), dtype="int32"):
        # from tvm.script import relax as R
        
        @R.function
        def nested(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            y: R.Tensor((), dtype="int32") = R.add(x, x)
            q: R.Tensor((), dtype="int32") = R.call_pure_packed("vm.builtin.copy", y, sinfo_args=(R.Tensor((), dtype="int32"),))
            return q

        z: R.Tensor((), dtype="int32") = R.const(1, "int32")
        w: R.Tensor((), dtype="int32") = nested(z)
        return w

    @R.function
    def use_call_pure_packed(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        y: R.Tensor((), dtype="int32") = R.add(x, x)
        z: R.Tensor((), dtype="int32") = R.call_pure_packed("vm.builtin.copy", y, sinfo_args=(R.Tensor((), dtype="int32"),))
        return z

    @R.function
    def use_invoke_pure_closure(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        cls = Module
        closure: R.Object = R.make_closure(cls.base, R.tuple())
        res: R.Tensor((), dtype="int32") = R.invoke_pure_closure(closure, (x,), sinfo_args=(R.Tensor((), dtype="int32"),))
        return res
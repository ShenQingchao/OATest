# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    extern_func = R.ExternFunc("extern_func")
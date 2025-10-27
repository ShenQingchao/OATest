# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def foo() -> R.Tensor((32, 64), dtype="int32"):
        obj: R.Object = R.vm.alloc_storage(R.shape([24576]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        a: R.Tensor((32, 64), dtype="int32") = R.vm.alloc_tensor(obj, R.prim_value(0), R.shape([32, 64]), R.dtype("int32"))
        ret_val: R.Tensor((32, 64), dtype="int32") = R.builtin.alloc_tensor(R.shape([32, 64]), R.dtype("int32"), R.prim_value(0), R.str("global"))
        R.vm.kill_object(a)
        R.vm.kill_object(obj)
        lv: R.Tensor((32, 64), dtype="int32") = ret_val
        return lv
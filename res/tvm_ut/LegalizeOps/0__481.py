# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def transform_params() -> R.Tuple:
        ic = T.int64()
        gv: R.Object = R.call_packed("get_item", R.prim_value(0), sinfo_args=(R.Object,))
        gv1: R.Tensor((3, ic, 3, 3), dtype="float32") = R.match_cast(gv, R.Tensor((3, ic, 3, 3), dtype="float32"))
        param0: R.Tensor((3, ic, 3, 3), dtype="float32") = gv1
        gv2: R.Object = R.call_packed("get_item", R.prim_value(1), sinfo_args=(R.Object,))
        gv3: R.Tensor((16, 16, 3, 3), dtype="float32") = R.match_cast(gv2, R.Tensor((16, 16, 3, 3), dtype="float32"))
        param1: R.Tensor((16, 16, 3, 3), dtype="float32") = gv3
        _: R.Object = R.call_packed("set_item", R.prim_value(1), param1, sinfo_args=(R.Object,))
        R.vm.kill_object(param1)
        transformed0: R.Tensor((ic, 3, 3, 3), dtype="float32") = R.permute_dims(param0, axes=[1, 0, 2, 3])
        R.vm.kill_object(param0)
        _3: R.Object = R.call_packed("set_item", R.prim_value(0), transformed0, sinfo_args=(R.Object,))
        R.tuple()
        return R.tuple()
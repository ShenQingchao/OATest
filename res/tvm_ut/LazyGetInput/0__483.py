# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def transform_params(rank_arg: R.Prim(value="rank"), world_size_arg: R.Prim(value="world_size"), weight_A: R.Tensor((16, 64), dtype="float32"), weight_B: R.Tensor((1024, 2048), dtype="float32")) -> R.Tuple(R.Tensor(("T.min((rank * 16 + 16) // world_size, 16) - T.min(rank * 16 // world_size, 16)", 64), dtype="float32"), R.Tensor((1024, "T.min((rank * 2048 + 2048) // world_size, 2048) - T.min(rank * 2048 // world_size, 2048)"), dtype="float32")):
        rank = T.int64()
        world_size = T.int64()
        R.func_attr({"num_input": 2})
        weight_A_1: R.Tensor((T.min((rank * 16 + 16) // world_size, 16) - T.min(rank * 16 // world_size, 16), 64), dtype="float32") = R.strided_slice(weight_A, (R.prim_value(0),), (R.prim_value(rank * 16 // world_size),), (R.prim_value((rank + 1) * 16 // world_size),), assume_inbound=False)
        weight_B_1: R.Tensor((1024, T.min((rank * 2048 + 2048) // world_size, 2048) - T.min(rank * 2048 // world_size, 2048)), dtype="float32") = R.strided_slice(weight_B, (R.prim_value(1),), (R.prim_value(rank * 2048 // world_size),), (R.prim_value((rank + 1) * 2048 // world_size),), assume_inbound=False)
        return (weight_A_1, weight_B_1)

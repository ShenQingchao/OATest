# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3,), dtype="int64")) -> R.Tensor((3,), dtype="int64"):
        x_1 = T.int64()
        gv: R.Shape([3]) = R.call_pure_packed("vm.builtin.tensor_to_shape", x, sinfo_args=(R.Shape([3]),))
        y: R.Shape([x_1]) = R.match_cast(gv, R.Shape([x_1]))
        lv: R.Shape([x_1]) = R.shape([x_1])
        gv_1: R.Tensor((x_1,), dtype="int64") = R.reshape(x, lv)
        return gv_1
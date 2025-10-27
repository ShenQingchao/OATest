# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def identity(rxplaceholder: T.Buffer((T.int64(3), T.int64(3)), "float32"), T_id: T.Buffer((T.int64(3), T.int64(3)), "float32")):
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(rxplaceholder[v_ax0, v_ax1])
                T.writes(T_id[v_ax0, v_ax1])
                T_id[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1]

    @R.function
    def mul2(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((3, 3), dtype="float32"):
        gv: R.Tensor((3, 3), dtype="float32") = R.multiply(x, R.const(2, "float32"))
        return gv

    @R.function
    def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((3, 3), dtype="float32"):
        cls = Module
        gv: R.Tensor((3, 3), dtype="float32") = cls.mul2(x)
        gv1 = R.call_tir(cls.identity, (gv,), out_sinfo=R.Tensor((3, 3), dtype="float32"))
        gv2: R.Tensor((3, 3), dtype="float32") = R.multiply(gv1, R.const(2, "float32"))
        return gv2
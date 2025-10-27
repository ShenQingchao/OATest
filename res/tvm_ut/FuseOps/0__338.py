# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(var_x: T.handle, B: T.Buffer((), "float32"), var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        x = T.match_buffer(var_x, (n, m))
        T_add = T.match_buffer(var_T_add, (n, m))
        # with T.block("root"):
        for ax0, ax1 in T.grid(n, m):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1], B[()])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = x[v_ax0, v_ax1] + B[()]

    @T.prim_func(private=True)
    def exp(var_lv0: T.handle, var_compute: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv0 = T.match_buffer(var_lv0, (n, m))
        compute = T.match_buffer(var_compute, (n, m))
        # with T.block("root"):
        for i0, i1 in T.grid(n, m):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv0[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.exp(lv0[v_i0, v_i1])

    @T.prim_func(private=True)
    def squeeze(var_lv1: T.handle, var_T_squeeze: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n, m = T.int64(), T.int64()
        lv1 = T.match_buffer(var_lv1, (n, m))
        T_squeeze = T.match_buffer(var_T_squeeze, (n, m))
        # with T.block("root"):
        for ax0, ax1 in T.grid(n, m):
            with T.block("T_squeeze"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[v_ax0, v_ax1])
                T.writes(T_squeeze[v_ax0, v_ax1])
                T_squeeze[v_ax0, v_ax1] = lv1[v_ax0, v_ax1]

    @R.function
    def main(x: R.Tensor(("n", "m"), dtype="float32")) -> R.Tensor(("n", "m"), dtype="float32"):
        n = T.int64()
        m = T.int64()
        cls = Module
        with R.dataflow():
            lv0 = R.call_tir(cls.add, (x, R.const(1, "float32")), out_sinfo=R.Tensor((n, m), dtype="float32"))
            lv1 = R.call_tir(cls.exp, (lv0,), out_sinfo=R.Tensor((n, m), dtype="float32"))
            gv = R.call_tir(cls.squeeze, (lv1,), out_sinfo=R.Tensor((n, m), dtype="float32"))
            R.output(gv)
        return gv
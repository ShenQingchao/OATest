# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(x: T.Buffer((T.int64(10), T.int64(20)), "float32"), B: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(10), T.int64(20)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1], B[()])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = x[v_ax0, v_ax1] + B[()]

    @T.prim_func(private=True)
    def add1(x: T.Buffer((T.int64(20), T.int64(10)), "float32"), B: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(20), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(20), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1], B[()])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = x[v_ax0, v_ax1] + B[()]

    @T.prim_func(private=True)
    def exp(lv: T.Buffer((T.int64(10), T.int64(20)), "float32"), compute: T.Buffer((T.int64(10), T.int64(20)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.exp(lv[v_i0, v_i1])

    @T.prim_func(private=True)
    def exp1(lv2: T.Buffer((T.int64(20), T.int64(10)), "float32"), compute: T.Buffer((T.int64(20), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(20), T.int64(10)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv2[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.exp(lv2[v_i0, v_i1])

    @T.prim_func(private=True)
    def squeeze(lv1: T.Buffer((T.int64(10), T.int64(20)), "float32"), T_squeeze: T.Buffer((T.int64(10), T.int64(20)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[v_ax0, v_ax1])
                T.writes(T_squeeze[v_ax0, v_ax1])
                T_squeeze[v_ax0, v_ax1] = lv1[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def squeeze1(lv3: T.Buffer((T.int64(20), T.int64(10)), "float32"), T_squeeze: T.Buffer((T.int64(20), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(20), T.int64(10)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv3[v_ax0, v_ax1])
                T.writes(T_squeeze[v_ax0, v_ax1])
                T_squeeze[v_ax0, v_ax1] = lv3[v_ax0, v_ax1]

    @R.function
    def func1(x: R.Tensor((10, 20), dtype="float32")) -> R.Tensor((10, 20), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.add, (x, R.const(1, "float32")), out_sinfo=R.Tensor((10, 20), dtype="float32"))
            lv1 = R.call_tir(cls.exp, (lv,), out_sinfo=R.Tensor((10, 20), dtype="float32"))
            gv = R.call_tir(cls.squeeze, (lv1,), out_sinfo=R.Tensor((10, 20), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def func2(x: R.Tensor((20, 10), dtype="float32")) -> R.Tensor((20, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv2 = R.call_tir(cls.add1, (x, R.const(1, "float32")), out_sinfo=R.Tensor((20, 10), dtype="float32"))
            lv3 = R.call_tir(cls.exp1, (lv2,), out_sinfo=R.Tensor((20, 10), dtype="float32"))
            gv1 = R.call_tir(cls.squeeze1, (lv3,), out_sinfo=R.Tensor((20, 10), dtype="float32"))
            R.output(gv1)
        return gv1
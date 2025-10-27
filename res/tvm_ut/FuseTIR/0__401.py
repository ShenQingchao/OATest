# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(var_lv0: T.handle, var_lv0_1: T.handle, var_T_add: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv0 = T.match_buffer(var_lv0, (T.int64(1), n, T.int64(1)))
        lv0_1 = T.match_buffer(var_lv0_1, (T.int64(1), n, T.int64(1)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(1)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(1)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv0[v_ax0, v_ax1, v_ax2], lv0_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = lv0[v_ax0, v_ax1, v_ax2] + lv0_1[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def divide(var_y: T.handle, var_lv2: T.handle, var_T_divide: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        y = T.match_buffer(var_y, (T.int64(1), n, T.int64(4096)))
        lv2 = T.match_buffer(var_lv2, (T.int64(1), n, T.int64(1)))
        T_divide = T.match_buffer(var_T_divide, (T.int64(1), n, T.int64(4096)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(y[v_ax0, v_ax1, v_ax2], lv2[v_ax0, v_ax1, T.int64(0)])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = y[v_ax0, v_ax1, v_ax2] / lv2[v_ax0, v_ax1, T.int64(0)]

    @T.prim_func(private=True)
    def multiply(rms_norm_weight: T.Buffer((T.int64(4096),), "float32"), var_lv3: T.handle, var_T_multiply: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv3 = T.match_buffer(var_lv3, (T.int64(1), n, T.int64(4096)))
        T_multiply = T.match_buffer(var_T_multiply, (T.int64(1), n, T.int64(4096)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(rms_norm_weight[v_ax2], lv3[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = rms_norm_weight[v_ax2] * lv3[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def sqrt(var_lv1: T.handle, var_compute: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv1 = T.match_buffer(var_lv1, (T.int64(1), n, T.int64(1)))
        compute = T.match_buffer(var_compute, (T.int64(1), n, T.int64(1)))
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(1)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(lv1[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.sqrt(lv1[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def te_mean(var_x: T.handle, var_T_divide: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        x = T.match_buffer(var_x, (T.int64(1), n, T.int64(4096)))
        T_divide = T.match_buffer(var_T_divide, (T.int64(1), n, T.int64(1)))
        # with T.block("root"):
        x_red = T.alloc_buffer((T.int64(1), n, T.int64(1)))
        for ax0, ax1, ax2, k2 in T.grid(T.int64(1), n, T.int64(1), T.int64(4096)):
            with T.block("x_red"):
                v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                T.reads(x[v_ax0, v_ax1, v_k2])
                T.writes(x_red[v_ax0, v_ax1, v_ax2])
                with T.init():
                    x_red[v_ax0, v_ax1, v_ax2] = T.float32(0)
                x_red[v_ax0, v_ax1, v_ax2] = x_red[v_ax0, v_ax1, v_ax2] + x[v_ax0, v_ax1, v_k2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(1)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(x_red[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = x_red[v_ax0, v_ax1, v_ax2] * T.float32(0.000244140625)

    @R.function
    def fused_mean_add_tir_sqrt_divide_multiply(x: R.Tensor((1, "n", 4096), dtype="float32"), y: R.Tensor((1, "n", 4096), dtype="float32"), rms_norm_weight: R.Tensor((4096,), dtype="float32")) -> R.Tensor((1, "n", 4096), dtype="float32"):
        n = T.int64()
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv0 = R.call_tir(cls.te_mean, (x,), out_sinfo=R.Tensor((1, n, 1), dtype="float32"))
            lv1 = R.call_tir(cls.add, (lv0, lv0), out_sinfo=R.Tensor((1, n, 1), dtype="float32"))
            lv2 = R.call_tir(cls.sqrt, (lv1,), out_sinfo=R.Tensor((1, n, 1), dtype="float32"))
            lv3 = R.call_tir(cls.divide, (y, lv2), out_sinfo=R.Tensor((1, n, 4096), dtype="float32"))
            gv = R.call_tir(cls.multiply, (rms_norm_weight, lv3), out_sinfo=R.Tensor((1, n, 4096), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((1, "n", 4096), dtype="float32"), y: R.Tensor((1, "n", 4096), dtype="float32"), rms_norm_weight: R.Tensor((4096,), dtype="float32")) -> R.Tensor((1, "n", 4096), dtype="float32"):
        n = T.int64()
        cls = Module
        with R.dataflow():
            gv: R.Tensor((1, n, 4096), dtype="float32") = cls.fused_mean_add_tir_sqrt_divide_multiply(x, y, rms_norm_weight)
            R.output(gv)
        return gv
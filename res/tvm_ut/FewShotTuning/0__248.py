# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv26: T.Buffer((T.int64(1), T.int64(3456), T.int64(2560)), "float16"), T_multiply: T.Buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_strided_slice_with_axes = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_divide = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_2 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        compute = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)))
        compute_1 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)))
        compute_2 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_3 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_add = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_5 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_divide_1 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_add_1 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_add_2 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_6 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_strided_slice_with_axes_1 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_7 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_strided_slice_with_axes"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv26[v_ax0, v_ax1, v_ax2 + T.int64(1280)])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2] = lv26[v_ax0, v_ax1, v_ax2 + T.int64(1280)]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2] * T.float16(0.70718232044198892)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_divide[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T_multiply_1[v_ax0, v_ax1, v_ax2] = T_divide[v_ax0, v_ax1, v_ax2] * T.float16(1.4140625)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_2[v_ax0, v_ax1, v_ax2])
                T_multiply_2[v_ax0, v_ax1, v_ax2] = T_multiply_1[v_ax0, v_ax1, v_ax2] * T.float16(0.70710678118654757)
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_multiply_2[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float32", T_multiply_2[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("compute_1"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(compute[v_i0, v_i1, v_i2])
                T.writes(compute_1[v_i0, v_i1, v_i2])
                compute_1[v_i0, v_i1, v_i2] = T.erf(compute[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("compute_2"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(compute_1[v_i0, v_i1, v_i2])
                T.writes(compute_2[v_i0, v_i1, v_i2])
                compute_2[v_i0, v_i1, v_i2] = T.Cast("float16", compute_1[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_1_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(compute_2[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_3[v_ax0, v_ax1, v_ax2])
                T_multiply_3[v_ax0, v_ax1, v_ax2] = compute_2[v_ax0, v_ax1, v_ax2] * T.float16(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_3[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = T.float16(0.5) + T_multiply_3[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2])
                T_multiply_4[v_ax0, v_ax1, v_ax2] = T_multiply_1[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_3"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_4[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_5[v_ax0, v_ax1, v_ax2])
                T_multiply_5[v_ax0, v_ax1, v_ax2] = T_multiply_4[v_ax0, v_ax1, v_ax2] * T.float16(1.4140625)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_divide_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_5[v_ax0, v_ax1, v_ax2], T_divide[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide_1[v_ax0, v_ax1, v_ax2])
                T_divide_1[v_ax0, v_ax1, v_ax2] = T_multiply_5[v_ax0, v_ax1, v_ax2] / T_divide[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_divide_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_1[v_ax0, v_ax1, v_ax2])
                T_add_1[v_ax0, v_ax1, v_ax2] = T_divide_1[v_ax0, v_ax1, v_ax2] + T.float16(-1)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_add_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_2[v_ax0, v_ax1, v_ax2])
                T_add_2[v_ax0, v_ax1, v_ax2] = T_add_1[v_ax0, v_ax1, v_ax2] + T.float16(1)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_4"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2], T_add_2[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_6[v_ax0, v_ax1, v_ax2])
                T_multiply_6[v_ax0, v_ax1, v_ax2] = T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2] * T_add_2[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_strided_slice_with_axes_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv26[v_ax0, v_ax1, v_ax2])
                T.writes(T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2])
                T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2] = lv26[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_5"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_6[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_7[v_ax0, v_ax1, v_ax2])
                T_multiply_7[v_ax0, v_ax1, v_ax2] = T_multiply_6[v_ax0, v_ax1, v_ax2] * T.float16(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_6"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2], T_multiply_7[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2] * T_multiply_7[v_ax0, v_ax1, v_ax2]
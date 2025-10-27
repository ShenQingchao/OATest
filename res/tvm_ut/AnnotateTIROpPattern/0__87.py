# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def max_pool2d(rxplaceholder_1: T.Buffer((1, 64, 112, 112), "float32"), tensor_1: T.Buffer((1, 64, 56, 56), "float32")):
        T.func_attr({"T.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp_1 = T.alloc_buffer((1, 64, 114, 114))
        for i0, i1, i2, i3 in T.grid(1, 64, 114, 114):
            with T.block("pad_temp"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(rxplaceholder_1[ax0, ax1, ax2 - 1, ax3 - 1])
                T.writes(pad_temp_1[ax0, ax1, ax2, ax3])
                pad_temp_1[ax0, ax1, ax2, ax3] = T.if_then_else(1 <= ax2 and ax2 < 113 and 1 <= ax3 and ax3 < 113, rxplaceholder_1[ax0, ax1, ax2 - 1, ax3 - 1], T.float32(-3.4028234663852886e+38))
        for i0, i1, i2, i3, i4, i5 in T.grid(1, 64, 56, 56, 3, 3):
            with T.block("tensor"):
                ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                T.reads(tensor_1[ax0, ax1, ax2, ax3], pad_temp_1[ax0, ax1, ax2 * 2 + rv0, ax3 * 2 + rv1])
                T.writes(tensor_1[ax0, ax1, ax2, ax3])
                with T.init():
                    tensor_1[ax0, ax1, ax2, ax3] = T.float32(-3.4028234663852886e+38)
                tensor_1[ax0, ax1, ax2, ax3] = T.max(tensor_1[ax0, ax1, ax2, ax3], pad_temp_1[ax0, ax1, ax2 * 2 + rv0, ax3 * 2 + rv1])
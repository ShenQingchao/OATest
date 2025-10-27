# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def softmax(rxplaceholder_1: T.Buffer((16, 16), "float32"), T_softmax_norm_1: T.Buffer((16, 16), "float32")):
        T.func_attr({"T.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem_1 = T.alloc_buffer((16,))
        T_softmax_exp_1 = T.alloc_buffer((16, 16))
        T_softmax_expsum_1 = T.alloc_buffer((16,))
        for i0_7, i1_3 in T.grid(16, 16):
            with T.block("T_softmax_maxelem"):
                i0_8, k = T.axis.remap("SR", [i0_7, i1_3])
                T.reads(T_softmax_maxelem_1[i0_8], rxplaceholder_1[i0_8, k])
                T.writes(T_softmax_maxelem_1[i0_8])
                with T.init():
                    T_softmax_maxelem_1[i0_8] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem_1[i0_8] = T.max(T_softmax_maxelem_1[i0_8], rxplaceholder_1[i0_8, k])
        for i0_9, i1_4 in T.grid(16, 16):
            with T.block("T_softmax_exp"):
                i0_10, i1_5 = T.axis.remap("SS", [i0_9, i1_4])
                T.reads(rxplaceholder_1[i0_10, i1_5], T_softmax_maxelem_1[i0_10])
                T.writes(T_softmax_exp_1[i0_10, i1_5])
                T_softmax_exp_1[i0_10, i1_5] = T.exp(rxplaceholder_1[i0_10, i1_5] - T_softmax_maxelem_1[i0_10])
        for i0_11, i1_6 in T.grid(16, 16):
            with T.block("T_softmax_expsum"):
                i0_12, k = T.axis.remap("SR", [i0_11, i1_6])
                T.reads(T_softmax_expsum_1[i0_12], T_softmax_exp_1[i0_12, k])
                T.writes(T_softmax_expsum_1[i0_12])
                with T.init():
                    T_softmax_expsum_1[i0_12] = T.float32(0)
                T_softmax_expsum_1[i0_12] = T_softmax_expsum_1[i0_12] + T_softmax_exp_1[i0_12, k]
        for i0_13, i1_7 in T.grid(16, 16):
            with T.block("T_softmax_norm"):
                i0_14, i1_8 = T.axis.remap("SS", [i0_13, i1_7])
                T.reads(T_softmax_exp_1[i0_14, i1_8], T_softmax_expsum_1[i0_14])
                T.writes(T_softmax_norm_1[i0_14, i1_8])
                T.block_attr({"axis": 1})
                T_softmax_norm_1[i0_14, i1_8] = T_softmax_exp_1[i0_14, i1_8] / T_softmax_expsum_1[i0_14]
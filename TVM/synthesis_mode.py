import tvm
import splice
import splice_graph

head_irs = """
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R
"""


def synthesize(base_irs, donor_irs, pass_level='dataflow'):
    if pass_level == 'block':
        synthesize_res = splice.insert_subgraph(base_irs, donor_irs)
    elif pass_level == 'dataflow':
        synthesize_res = splice_graph.insert_subgraph(base_irs, donor_irs)
    else:
        assert False, f"Cannot identify the CG level {pass_level}"
    return synthesize_res


if __name__ == '__main__':

    base_irs = """# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def cast(v2_0: T.Buffer((T.int64(25), T.int64(58)), "int32"), compute: T.Buffer((T.int64(25), T.int64(58)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(25), T.int64(58)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(v2_0[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = v2_0[v_i0, v_i1]

    @T.prim_func(private=True)
    def concatenate(lv: T.Buffer((T.int64(1), T.int64(1)), "int32"), lv1: T.Buffer((T.int64(1), T.int64(1)), "int32"), T_concat: T.Buffer((T.int64(2), T.int64(1)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(2), T.int64(1)):
            with T.block("T_concat"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[v_ax0 - T.int64(1), v_ax1], lv[v_ax0, v_ax1])
                T.writes(T_concat[v_ax0, v_ax1])
                T_concat[v_ax0, v_ax1] = T.if_then_else(T.int64(1) <= v_ax0, lv1[v_ax0 - T.int64(1), v_ax1], lv[v_ax0, v_ax1])

    @T.prim_func(private=True)
    def concatenate1(v5_0: T.Buffer((T.int64(1),), "int32"), v6_0: T.Buffer((T.int64(1),), "int32"), v7_0: T.Buffer((T.int64(1),), "int32"), v8_0: T.Buffer((T.int64(55),), "int32"), T_concat: T.Buffer((T.int64(58),), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(58)):
            with T.block("T_concat"):
                v_ax0 = T.axis.spatial(T.int64(58), ax0)
                T.reads(v8_0[v_ax0 - T.int64(3)], v7_0[v_ax0 - T.int64(2)], v6_0[v_ax0 - T.int64(1)], v5_0[v_ax0])
                T.writes(T_concat[v_ax0])
                T_concat[v_ax0] = T.if_then_else(T.int64(3) <= v_ax0, v8_0[v_ax0 - T.int64(3)], T.if_then_else(T.int64(2) <= v_ax0, v7_0[v_ax0 - T.int64(2)], T.if_then_else(T.int64(1) <= v_ax0, v6_0[v_ax0 - T.int64(1)], v5_0[v_ax0])))

    @T.prim_func(private=True)
    def expand_dims(v7_0: T.Buffer((T.int64(1),), "int32"), expand_dims: T.Buffer((T.int64(1), T.int64(1)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(1)):
            with T.block("expand_dims"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(v7_0[v_i1])
                T.writes(expand_dims[v_i0, v_i1])
                expand_dims[v_i0, v_i1] = v7_0[v_i1]

    @T.prim_func(private=True)
    def less(lv5: T.Buffer((T.int64(25), T.int64(58)), "int32"), lv4: T.Buffer((T.int64(58),), "int32"), T_less: T.Buffer((T.int64(25), T.int64(58)), "bool")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(25), T.int64(58)):
            with T.block("T_less"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv5[v_ax0, v_ax1], lv4[v_ax1])
                T.writes(T_less[v_ax0, v_ax1])
                T_less[v_ax0, v_ax1] = lv5[v_ax0, v_ax1] < lv4[v_ax1]

    @T.prim_func(private=True)
    def max(lv2: T.Buffer((T.int64(2), T.int64(1)), "int32"), lv2_red: T.Buffer((T.int64(1),), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, k0 in T.grid(T.int64(1), T.int64(2)):
            with T.block("lv2_red"):
                v_ax0, v_k0 = T.axis.remap("SR", [ax0, k0])
                T.reads(lv2[v_k0, v_ax0])
                T.writes(lv2_red[v_ax0])
                with T.init():
                    lv2_red[v_ax0] = -2147483648
                lv2_red[v_ax0] = T.max(lv2_red[v_ax0], lv2[v_k0, v_ax0])

    @T.prim_func(private=True)
    def where(v1_0: T.Buffer((), "bool"), lv3: T.Buffer((T.int64(25), T.int64(58)), "int32"), lv4: T.Buffer((T.int64(58),), "int32"), T_where: T.Buffer((T.int64(25), T.int64(58)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(25), T.int64(58)):
            with T.block("T_where"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(v1_0[()], lv3[v_ax0, v_ax1], lv4[v_ax1])
                T.writes(T_where[v_ax0, v_ax1])
                T_where[v_ax0, v_ax1] = T.Select(T.int64(0) < T.Cast("int64", v1_0[()]), lv3[v_ax0, v_ax1], lv4[v_ax1])

    @R.function(private=True)
    def main(v5_0: R.Tensor((1,), dtype="int32"), v1_0: R.Tensor((), dtype="bool"), v8_0: R.Tensor((55,), dtype="int32"), v7_0: R.Tensor((1,), dtype="int32"), v6_0: R.Tensor((1,), dtype="int32"), v2_0: R.Tensor((25, 58), dtype="int32")) -> R.Tuple(R.Tensor((1,), dtype="int32"), R.Tensor((25, 58), dtype="int32"), R.Tensor((25, 58), dtype="bool")):
        R.func_attr({"num_input": 2})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.expand_dims, (v7_0,), out_sinfo=R.Tensor((1, 1), dtype="int32"))
            lv1 = R.call_tir(cls.expand_dims, (v7_0,), out_sinfo=R.Tensor((1, 1), dtype="int32"))
            lv2 = R.call_tir(cls.concatenate, (lv, lv1), out_sinfo=R.Tensor((2, 1), dtype="int32"))
            lv3 = R.call_tir(cls.cast, (v2_0,), out_sinfo=R.Tensor((25, 58), dtype="int32"))
            lv4 = R.call_tir(cls.concatenate1, (v5_0, v6_0, v7_0, v8_0), out_sinfo=R.Tensor((58,), dtype="int32"))
            lv5 = R.call_tir(cls.where, (v1_0, lv3, lv4), out_sinfo=R.Tensor((25, 58), dtype="int32"))
            lv6 = R.call_tir(cls.max, (lv2,), out_sinfo=R.Tensor((1,), dtype="int32"))
            lv7 = R.call_tir(cls.less, (lv5, lv4), out_sinfo=R.Tensor((25, 58), dtype="bool"))
            gv: R.Tuple(R.Tensor((1,), dtype="int32"), R.Tensor((25, 58), dtype="int32"), R.Tensor((25, 58), dtype="bool")) = lv6, lv3, lv7
            R.output(gv)
        return gv
    @R.function(private=True)
    def main2(v5_0: R.Tensor((1,), dtype="int32"), v1_0: R.Tensor((), dtype="bool"), v8_0: R.Tensor((55,), dtype="int32"), v7_0: R.Tensor((1,), dtype="int32"), v6_0: R.Tensor((1,), dtype="int32"), v2_0: R.Tensor((25, 58), dtype="int32")) -> R.Tuple(R.Tensor((1,), dtype="int32"), R.Tensor((25, 58), dtype="int32"), R.Tensor((25, 58), dtype="bool")):
        R.func_attr({"num_input": 2})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.expand_dims, (v7_0,), out_sinfo=R.Tensor((1, 1), dtype="int32"))
            lv1 = R.call_tir(cls.expand_dims, (v7_0,), out_sinfo=R.Tensor((1, 1), dtype="int32"))
            lv2 = R.call_tir(cls.concatenate, (lv, lv1), out_sinfo=R.Tensor((2, 1), dtype="int32"))
            lv3 = R.call_tir(cls.cast, (v2_0,), out_sinfo=R.Tensor((25, 58), dtype="int32"))
            lv4 = R.call_tir(cls.concatenate1, (v5_0, v6_0, v7_0, v8_0), out_sinfo=R.Tensor((58,), dtype="int32"))
            lv5 = R.call_tir(cls.where, (v1_0, lv3, lv4), out_sinfo=R.Tensor((25, 58), dtype="int32"))
            lv6 = R.call_tir(cls.max, (lv2,), out_sinfo=R.Tensor((1,), dtype="int32"))
            lv7 = R.call_tir(cls.less, (lv5, lv4), out_sinfo=R.Tensor((25, 58), dtype="bool"))
            gv: R.Tuple(R.Tensor((1,), dtype="int32"), R.Tensor((25, 58), dtype="int32"), R.Tensor((25, 58), dtype="bool")) = lv6, lv3, lv7
            R.output(gv)
        return gv
"""
    donor_irs = """# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def relax_reshape_replacement(A: T.Buffer((T.int64(850), T.int64(2), T.int64(1024)), "float16"), T_reshape: T.Buffer((T.int64(850), T.int64(1), T.int64(2048)), "float16")):
        T.func_attr({"operator_name": "relax.reshape"})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(850), T.int64(1), T.int64(2048)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax2 // T.int64(1024), v_ax2 % T.int64(1024)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax2 // T.int64(1024), v_ax2 % T.int64(1024)]

    @T.prim_func(private=True)
    def reshape(A: T.Buffer((T.int64(850), T.int64(2048)), "float16"), T_reshape: T.Buffer((T.int64(850), T.int64(1), T.int64(2048)), "float16")):
        T.func_attr({"operator_name": "relax.reshape"})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(850), T.int64(1), T.int64(2048)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[(v_ax2 // T.int64(2048) + v_ax0 + v_ax1) % T.int64(850), v_ax2 % T.int64(2048)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax2 // T.int64(2048) + v_ax0 + v_ax1) % T.int64(850), v_ax2 % T.int64(2048)]

    @R.function
    def main1000(x: R.Tensor((850, 2048), dtype="float16")) -> R.Tensor((850, 1, 2048), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((850, 2, 1024), dtype="float16") = R.layout_transform(x, index_map=T.index_map(lambda i, j: (i, j // 1024, j % 1024)), pad_value=None, axis_separators=[])
            lv_1 = R.call_tir(cls.relax_reshape_replacement, (lv,), out_sinfo=R.Tensor((850, 1, 2048), dtype="float16"))
            gv: R.Tensor((850, 1, 2048), dtype="float16") = lv_1
            R.output(gv)
        return gv

"""
    base_irs = tvm.script.from_source(base_irs)
    from tvm import relax
    base_irs = relax.transform.LegalizeOps()(base_irs)  # correct the func return para shape
    print('*'*1000)
    base_irs.show()

    donor_irs = tvm.script.from_source(donor_irs)
    donor_irs.show()
    print("Synthesized model:")
    new_irs = synthesize(base_irs, donor_irs)
    # print(new_irs)
    print("finish ALL!")

# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def sum(rxplaceholder: T.Buffer((T.int64(3), T.int64(3)), "float32"), rxplaceholder_red: T.Buffer((), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for k0, k1 in T.grid(T.int64(3), T.int64(3)):
            with T.block("rxplaceholder_red"):
                v_k0, v_k1 = T.axis.remap("RR", [k0, k1])
                T.reads(rxplaceholder[v_k0, v_k1])
                T.writes(rxplaceholder_red[()])
                with T.init():
                    rxplaceholder_red[()] = T.float32(0)
                rxplaceholder_red[()] = rxplaceholder_red[()] + rxplaceholder[v_k0, v_k1]

    @R.function
    def main(x0: R.Tensor((3, 3), dtype="float32"), x1: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.sum(x0, axis=None, keepdims=False)
            R.output(gv)
        return gv
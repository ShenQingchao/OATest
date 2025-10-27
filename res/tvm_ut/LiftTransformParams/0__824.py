# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def slice(Input_2d: T.Buffer((16, 16), "int32"), Output_Slice: T.Buffer((16,), "int32"), slice_index: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for j in range(16):
            with T.block("T_full"):
                vj = T.axis.spatial(16, j)
                T.reads(Input_2d[slice_index, vj])
                T.writes(Output_Slice[vj])
                Output_Slice[vj] = Input_2d[slice_index, vj]

    @R.function
    def main(A: R.Tensor((16, 16), dtype="int32"), B: R.Tensor((16, 16), dtype="int32"), shape: R.Shape(["slice_index"])) -> R.Tensor((16,), dtype="int32"):
        slice_index = T.int64()
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            B_slice = R.call_tir(cls.slice, (B,), out_sinfo=R.Tensor((16,), dtype="int32"), tir_vars=R.shape([slice_index]))
            A_slice = R.call_tir(cls.slice, (A,), out_sinfo=R.Tensor((16,), dtype="int32"), tir_vars=R.shape([slice_index]))
            A_scale: R.Tensor((16,), dtype="int32") = R.multiply(A_slice, B_slice)
            R.output(A_scale)
        return A_scale
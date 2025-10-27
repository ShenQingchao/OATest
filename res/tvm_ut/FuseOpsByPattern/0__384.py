# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1024, 1024), dtype="float16"), w: R.Tensor((1024, 1024), dtype="float16"), transpose_weights: R.Prim("bool")) -> R.Tensor((1024, 1024), dtype="float16"):
        if transpose_weights:
            with R.dataflow():
                w_t: R.Tensor((1024, 1024), dtype="float16") = R.permute_dims(w, axes=None)
                out_then: R.Tensor((1024, 1024), dtype="float16") = R.matmul(x, w_t, out_dtype="void")
                R.output(out_then)
            out: R.Tensor((1024, 1024), dtype="float16") = out_then
        else:
            with R.dataflow():
                out_else: R.Tensor((1024, 1024), dtype="float16") = R.matmul(x, w, out_dtype="void")
                R.output(out_else)
            out: R.Tensor((1024, 1024), dtype="float16") = out_else
        return out
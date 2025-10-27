# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((3, 10), dtype="float32"), w0: R.Tensor((10, 5), dtype="float32"), b0: R.Tensor((5,), dtype="float32"), label: R.Tensor((3, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv0: R.Tensor((3, 5), dtype="float32") = R.matmul(x, w0, out_dtype="void")
            out: R.Tensor((3, 5), dtype="float32") = R.add(lv0, b0)
            logits: R.Tensor((3, 5), dtype="float32") = R.nn.log_softmax(out, axis=-1)
            loss: R.Tensor((), dtype="float32") = R.nn.cross_entropy_with_logits(logits, label)
            R.output(loss)
        return loss
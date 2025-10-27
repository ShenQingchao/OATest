# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((640, 640), dtype="float32"), y: R.Tensor((640, 640), dtype="float32"), bias: R.Tensor((640,), dtype="float32"), y_1: R.Tensor((640, 640), dtype="float32"), y_2: R.Tensor((640, 640), dtype="float32"), bias_1: R.Tensor((640,), dtype="float32")) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((640, 640), dtype="float32") = R.matmul(x, y, out_dtype="float32")
            lv1: R.Tensor((640, 640), dtype="float32") = R.add(lv, bias)
            lv2: R.Tensor((640, 640), dtype="float32") = R.nn.relu(lv1)
            lv3: R.Tensor((640, 640), dtype="float32") = R.matmul(x, y_1, out_dtype="float32")
            lv4: R.Tensor((640, 640), dtype="float32") = R.matmul(x, y_2, out_dtype="float32")
            lv5: R.Tensor((640, 640), dtype="float32") = R.add(lv4, bias_1)
            lv6: R.Tensor((640, 640), dtype="float32") = R.nn.relu(lv5)
            lv7: R.Tensor((640, 1920), dtype="float32") = R.concat((lv2, lv3, lv6), axis=1)
            R.output(lv7)
        return lv7
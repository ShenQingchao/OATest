# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"attr": 10})
    I.module_global_infos({"vdevice": [I.vdevice({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-unknown-linux-gnu", "tag": ""}, 0, "global")]})
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), z: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32", vdevice="llvm:0"):
        x1: R.Tensor((2, 3), dtype="float32") = x
        y1: R.Tensor((2, 3), dtype="float32") = y
        x2: R.Tensor((2, 3), dtype="float32") = x1
        y2: R.Tensor((2, 3), dtype="float32") = y1
        s: R.Tensor((2, 3), dtype="float32", vdevice="llvm:0") = R.add(x2, y2)
        m: R.Tensor((2, 3), dtype="float32", vdevice="llvm:0") = R.multiply(s, z)
        return m
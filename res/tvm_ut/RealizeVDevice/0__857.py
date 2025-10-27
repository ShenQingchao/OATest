# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"attr": 10})
    I.module_global_infos({"vdevice": [I.vdevice({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}, 0, "global")]})
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), z: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32", vdevice="cuda:0"):
        with R.dataflow():
            lv0: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
            gv: R.Tensor((2, 3), dtype="float32") = R.multiply(lv0, z)
            R.output(gv)
        return gv
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"attr": 10})
    I.module_global_infos({"vdevice": [I.vdevice({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-unknown-linux-gnu", "tag": ""}, 0, "global"), I.vdevice({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}, 0, "global"), I.vdevice({"keys": ["metal", "gpu"], "kind": "metal", "max_function_args": 31, "max_num_threads": 256, "max_shared_memory_per_block": 32768, "max_threads_per_block": 256, "tag": "", "thread_warp_size": 16}, 0, "global"), I.vdevice({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}, 0, "global")]})
    @R.function
    def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), z: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        with R.dataflow():
            lv0: R.Tensor((2, 3), dtype="float32") = R.hint_on_device(y, R.device(dev_type=1, dev_id=0))
            lv1: R.Tensor((2, 3), dtype="float32") = R.add(x, lv0)
            lv2: R.Tensor((2, 3), dtype="float32") = R.hint_on_device(lv1, R.device(dev_type=2, dev_id=0))
            lv3: R.Tensor((2, 3), dtype="float32") = R.add(lv2, lv2)
            lv4: R.Tensor((2, 3), dtype="float32") = R.hint_on_device(z, R.device(dev_type=2, dev_id=0))
            gv: R.Tensor((2, 3), dtype="float32") = R.multiply(lv3, lv4)
            R.output(gv)
        return gv
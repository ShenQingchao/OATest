# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 5), dtype="float32"):
        gv0: R.Tensor((10, 5), dtype="float32") = R.dist.redistribute_replica_to_shard(x, num_workers=2, axis=1)
        return gv0
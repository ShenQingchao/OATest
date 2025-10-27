# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main() -> R.Tuple(R.Object):
        _io: R.Object = R.null_value()
        lv: R.Tuple(R.Object) = (_io,)
        gv: R.Tuple(R.Object) = lv
        return gv
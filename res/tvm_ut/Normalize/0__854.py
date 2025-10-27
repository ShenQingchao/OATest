# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False, private=True)
    def main() -> R.Tuple:
        x: R.Tuple = R.assert_op(R.const(True, "bool"), format=R.str(""))
        return x
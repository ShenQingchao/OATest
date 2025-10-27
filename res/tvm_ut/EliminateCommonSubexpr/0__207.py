# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def foo(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        with R.dataflow():
            # from tvm.script import relax as R
            
            @R.function
            def bar(y: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
                with R.dataflow():
                    lv0: R.Tensor((), dtype="int32") = R.add(y, y)
                    lv1: R.Tensor((), dtype="int32") = R.add(y, y)
                    lv2: R.Tensor((), dtype="int32") = R.add(lv0, lv1)
                    gv: R.Tensor((), dtype="int32") = lv2
                    R.output(gv)
                gv_1: R.Tensor((), dtype="int32") = R.add(gv, gv)
                return gv_1

            lv0: R.Tensor((), dtype="int32") = bar(x)
            lv1: R.Tensor((), dtype="int32") = bar(x)
            lv2: R.Tensor((), dtype="int32") = R.add(lv0, lv1)
            lv3: R.Tensor((), dtype="int32") = bar(x)
            lv4: R.Tensor((), dtype="int32") = bar(x)
            lv5: R.Tensor((), dtype="int32") = R.add(lv3, lv4)
            lv6: R.Tensor((), dtype="int32") = R.add(lv2, lv5)
            gv: R.Tensor((), dtype="int32") = lv6
            R.output(gv)
        return gv
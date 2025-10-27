import tvm
import re
from tvm.script import relax


def get_donor_inputs_types(fun_ir, func_input_constraints):
    def _get_op_class(op_name):
        if not op_name.startswith("relax"):
            return None
        op = eval(op_name)
        op_clazz = op.__module__

        op_clazz = op_clazz[13:]
        return op_clazz

    donor_flag = 0
    for block in fun_ir.body.blocks:
        bindings = block.bindings
        for bind in bindings:
            var_name = bind.var
            call = bind.value

            # import pdb;pdb.set_trace()
            if isinstance(call, tvm.relax.expr.Function):  # subfunc
                # import pdb;pdb.set_trace()
                donor_flag = max(donor_flag, get_donor_inputs_types(call, func_input_constraints))

            elif hasattr(call, "op") and hasattr(call.op, "name"):
                op_name = call.op.name
                arg_list = call.args
                arg_list = (str(v) for v in arg_list)
                print('func_input_constraints', func_input_constraints)
                undefined_var_dict = {k: v for k, v, t in func_input_constraints}
                undefined_var_name = undefined_var_dict.keys()
                undefined_var_num = len(set(arg_list).intersection(set(undefined_var_name)))
                #
                if undefined_var_num >= 1:
                    op_class = _get_op_class(op_name)
                    if op_class in ["unary", "create", "datatype", 'grad', 'index', 'inspect',
                                    'search', 'set', 'sorting']:
                        pass
                    elif op_class in ["binary", "ternary", ]:
                        donor_flag = max(donor_flag, 1)
                    else:
                        donor_flag = max(donor_flag, 2)
            else:
                donor_flag = 2   # e.g., inner_func
    return donor_flag



def get_op_constraint_type(op_name, arg_set:set, undefined_var, all_before_vars=None):
    '''
    目的：将undefined var中已经有的shape抽象出来，并从used_var中选取可兼容的var
    约束：
    - 1个undefined var本身的约束
    - 两个undefined vars之间可能存在约束（比如，shape相等,shape中某个dim是相等的）
    解决方案：
    * 【确定ndim】首先根据donor中function的inputs的约束，确定ndim
    * 【创建符号】将每个undefined var表示为一个ndim大小的list。例如x=[x0,x1,x2,x3] 或y=[y0,y1]
    * 【归纳推理约束】推理原始有具体值的inputs中的“相等”因素。比如 x1=y1, x2=y2;
    * 【提取约束】然后根据op的约束，判断undefined var的shape值存在哪些约束，
    * 【增加基本约束】数值的取值范围来自used_var中的值。从used_var中挑选一个符合第一个var的值，然后再根据已有的约束，获取第二的var，依次类推。
    * 【补充约束】如果unused_var中并不存在符合约束的var,那么就要使用reshape了哈。
    '''

    donor_para_name, donor_para_shape, donor_para_dtype = undefined_var
    pass


def abs_input_constraints(fun_ir, func_input_constraints):
    pass


if __name__ == '__main__':

    from tvm.script import ir as I
    from tvm.script import relax as R

    @I.ir_module
    class Module:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(R.Tensor((2, 3), dtype="float32"),
                                                                 R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                a: R.Tensor((2, 3), dtype="float32") = R.exp(x)
                b: R.Tensor((2, 3), dtype="float32") = R.exp(a)
                c: R.Tensor((2, 3), dtype="float32") = R.nn.softmax(b, axis=-1)
                d = R.add(x,y)
                R.output(d)
            return d

    func= Module['foo']
    inputs = get_func_inputs_constraints(func)
    print(inputs)
    res = abs_input_constraints(func, inputs)
    print(res)

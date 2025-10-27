import re


def extract_function_body(code: str, indent='') -> str:
    pattern = re.compile(rf"""
        @R\.function.*?\s+                     
        def\s+\w+\(.*?\):\n?                
        (?:\s*R\.func_attr\(.*?\))?\s*?     # optional
        (?:\s*with\s+R\.dataflow\(\):)?     # optional
        (?P<body>.*)                        # save this
        (?:R\.output\(.*?\))?               # can not save R.output 
        (?:return\s+.*?$)                   # can not save the return
        """, re.DOTALL | re.VERBOSE)

    match = pattern.search(code)
    if match:
        body = match.group('body')
        body = re.sub(r'^\s*#.*\n?', '', body, flags=re.MULTILINE)
        body = re.sub(r'^\s*\n', '', body, flags=re.MULTILINE)
        body = re.sub(r'^\s*R\.output\(.*\n', '', body, flags=re.MULTILINE)  # remove R.output
        body = re.sub(r':\s*R\.\w+\(.*?\) = ', '=', body)  # remove the inferred type
        body = body.rstrip()
        # print(body)

        if len(body.splitlines()) >= 1:
            last_line = body.splitlines()[-1]
            indent_num = len(last_line) - len(last_line.strip())
            if "R.dataflow()" in body and "@R.function" not in body:
                tmp = body.split("with R.dataflow():\n")
                body_before, body_after = tmp[0], tmp[-1]
                body_b = "\n".join(line[indent_num-4:] for line in body_before.splitlines())
                body_a = "\n".join(line[indent_num:] for line in body_after.splitlines())
                body = body_b + body_a
                body = "\n".join(f"{indent}" + line for line in body.splitlines())
            else:
                print(indent_num)
                body = "\n".join(line[indent_num:] for line in body.splitlines())
                body = "\n".join(f"{indent}" + line for line in body.splitlines())
            return body
    print(f"[Warning] cannot extract donor body from {code}")
    return ""


if __name__ == '__main__':

    # 示例代码字符串
    code_str = """
    @R.function
    def main_7(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")) -> R.Tensor((2, 26, 26, 4), dtype="float16"):
        with R.dataflow():
            lv0: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
            lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
            lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(lv0, lv1, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC", out_dtype="void")
            R.output(lv2)
        gv3: R.Tensor((2, 26, 26, 4), dtype="float16") = R.astype(lv2, dtype="float16")
        return gv3
    
    """
    code_str = """
    @R.function
    def main_3(x: R.Tensor(("batch_size", 1024), dtype="float16"), w1: R.Tensor((1024, 1024), dtype="float16"), w2: R.Tensor((1024, "M"), dtype="float16")) -> R.Tuple(R.Tensor(("batch_size", 1024), dtype="float16"), R.Tensor(("batch_size", "M"), dtype="float16")):
        batch_size = T.int64()
        M = T.int64()
        with R.dataflow():
            matmul1: R.Tensor((batch_size, 1024), dtype="float16") = R.matmul(x, w1, out_dtype="void")
            matmul2: R.Tensor((batch_size, M), dtype="float16") = R.matmul(x, w2, out_dtype="void")
            out: R.Tuple(R.Tensor((batch_size, 1024), dtype="float16"), R.Tensor((batch_size, M), dtype="float16")) = matmul1, matmul2
            R.output(out)
        return out
    """

    code_str = """
    @R.function
    def unrelated_function(A: R.Tensor((16, 16), dtype="float16")) -> R.Tensor((16, 16), dtype="float16"):
        # from tvm.script import relax as R
    
        @R.function
        def inner_func75(B: R.Tensor((16, 16), dtype="float16")) -> R.Tensor((16, 16), dtype="float16"):
            with R.dataflow():
                C: R.Tensor((16, 16), dtype="float16") = R.multiply(B, R.const(2, "float16"))
                R.output(C)
            return C
    
        D: R.Tensor((16, 16), dtype="float16") = inner_func75(A)
        return D
    """

    #
    # from tvm.script import ir as I
    # from tvm.script import tir as T
    # from tvm.script import relax as R
    #
    # @I.ir_module
    # class Module:
    #     @R.function
    #     def main(x1: R.Tensor((10, 5), dtype="float32"), y1: R.Tensor((10, 5), dtype="float32")) -> R.Tensor((10, 5),
    #                                                                                                          dtype="float32"):
    #         n = T.int64()
    #         m = T.int64()
    #
    #         # from tvm.script import tir as T
    #         # from tvm.script import relax as R
    #
    #         @R.function
    #         def inner(x2: R.Tensor((n, m), dtype="float32"), y2: R.Tensor((n, m), dtype="float32")) -> R.Tensor((n, m),
    #                                                                                                             dtype="float32"):
    #             sum_inner: R.Tensor((n, m), dtype="float32") = R.add(x2, y2)
    #             return sum_inner
    #
    #         sum_main: R.Tensor((10, 5), dtype="float32") = inner(x1, y1)
    #         return sum_main
    #
    #

    # from tvm.script import ir as I
    # from tvm.script import relax as R
    #
    # @I.ir_module
    # class Module:
    #     @R.function
    #     def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
    #         R.func_attr({"relax.force_pure": 1})
    #         alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 3]), R.dtype("float32"),
    #                                                                           R.prim_value(0), R.str("global"))
    #         return x


    # # # 提取函数体
    # code_str = Module['main'].script()
    print(code_str)
    result = extract_function_body(code_str, indent="    ")
    print('results:')
    print(result)

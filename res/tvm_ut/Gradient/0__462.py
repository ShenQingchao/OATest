metadata = tvm.ir.load_json("""{
  \"root\": 1, 
  \"nodes\": [
    {
      \"type_key\": \"\"
    }, 
    {
      \"type_key\": \"Map\", 
      \"keys\": [
        \"relax.expr.Constant\"
      ], 
      \"data\": [2]
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [3]
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"10\", 
        \"data\": \"0\", 
        \"span\": \"0\", 
        \"struct_info_\": \"4\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"1\", 
        \"shape\": \"5\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"9\", 
        \"span\": \"0\", 
        \"struct_info_\": \"8\", 
        \"values\": \"6\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [7]
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"6\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"1\", 
        \"span\": \"0\", 
        \"values\": \"6\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeType\", 
      \"attrs\": {
        \"ndim\": \"1\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.DynTensorType\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"1\", 
        \"span\": \"0\"
      }
    }
  ], 
  \"b64ndarrays\": [
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAIgAQAGAAAAAAAAABgAAAAAAAAAAACAPwAAgD8AAIA/AACAPwAAgD8AAIA/\"
  ], 
  \"attrs\": {\"tvm_version\": \"0.17.dev0\"}
}""")
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((6,), dtype="float32"), y: R.Tensor((6, 3, 4), dtype="float32")) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv1: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.split(x, indices_or_sections=2, axis=0)
            lv2: R.Tensor((3,), dtype="float32") = lv1[0]
            lv3: R.Tensor((3,), dtype="float32") = lv1[1]
            lv4: R.Tensor((3,), dtype="float32") = R.add(lv2, lv3)
            lv5: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = lv4, lv3
            lv6: R.Tensor((6,), dtype="float32") = R.concat(lv5, axis=0)
            lv7: R.Tuple(R.Tensor((6,), dtype="float32"), R.Tensor((6,), dtype="float32")) = x, x
            lv8: R.Tensor((12,), dtype="float32") = R.concat(lv7, axis=0)
            lv9: R.Tensor((12,), dtype="float32") = R.concat(lv7, axis=0)
            lv10: R.Tensor((12,), dtype="float32") = R.add(lv8, lv9)
            lv11: R.Tuple(R.Tensor((6,), dtype="float32"), R.Tensor((6,), dtype="float32")) = R.split(lv10, indices_or_sections=2, axis=0)
            lv11_1: R.Tensor((6,), dtype="float32") = lv11[0]
            lv12: R.Tensor((6,), dtype="float32") = R.add(lv6, lv11_1)
            lv13: R.Tensor((6,), dtype="float32") = metadata["relax.expr.Constant"][0]
            lv14: R.Tensor((6,), dtype="float32") = R.add(lv12, lv13)
            lv15: R.Tensor((6,), dtype="float32") = R.subtract(lv13, lv14)
            lv16: R.Tensor((6,), dtype="float32") = R.multiply(lv14, lv15)
            lv17: R.Tensor((6,), dtype="float32") = R.multiply(lv15, lv16)
            lv18: R.Tensor((6,), dtype="float32") = R.tanh(lv17)
            lv19: R.Tensor((6,), dtype="float32") = R.sigmoid(lv18)
            lv20: R.Tensor((6, 4, 3), dtype="float32") = R.permute_dims(y, axes=[0, 2, 1])
            lv21: R.Tensor((6, 4, 3), dtype="float32") = R.sigmoid(lv20)
            lv22: R.Tensor((6, 3, 3), dtype="float32") = R.matmul(y, lv21, out_dtype="void")
            lv23: R.Tensor((6,), dtype="float32") = R.sum(lv22, axis=[1, 2], keepdims=False)
            lv24: R.Tensor((6,), dtype="float32") = R.add(lv19, lv23)
            lv25: R.Tensor((6,), dtype="float32") = R.nn.log_softmax(lv24, axis=-1)
            gv: R.Tensor((), dtype="float32") = R.nn.nll_loss(lv25, R.const(3, "int64"), reduction="mean", ignore_index=-100)
            R.output(gv)
        return gv
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
      \"data\": [3, 11]
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
        \"dtype\": \"int64\", 
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
        \"value\": \"2\"
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
        \"dtype\": \"int64\", 
        \"ndim\": \"1\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.Constant\", 
      \"attrs\": {
        \"_checked_type_\": \"18\", 
        \"data\": \"1\", 
        \"span\": \"0\", 
        \"struct_info_\": \"12\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"ndim\": \"1\", 
        \"shape\": \"13\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"17\", 
        \"span\": \"0\", 
        \"struct_info_\": \"16\", 
        \"values\": \"14\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [15]
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"2\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"1\", 
        \"span\": \"0\", 
        \"values\": \"14\"
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
        \"dtype\": \"int64\", 
        \"ndim\": \"1\", 
        \"span\": \"0\"
      }
    }
  ], 
  \"b64ndarrays\": [
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAABAAQACAAAAAAAAABAAAAAAAAAACAAAAAAAAAAIAAAAAAAAAA==\", 
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAABAAQACAAAAAAAAABAAAAAAAAAAAQAAAAAAAAABAAAAAAAAAA==\"
  ], 
  \"attrs\": {\"tvm_version\": \"0.17.dev0\"}
}""")
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((256,), dtype="float32")) -> R.Tensor(dtype="float32", ndim=2):
        with R.dataflow():
            lv0: R.Tensor((2,), dtype="int64") = R.add(metadata["relax.expr.Constant"][0], metadata["relax.expr.Constant"][0])
            target_shape: R.Tensor((2,), dtype="int64") = R.multiply(lv0, metadata["relax.expr.Constant"][1])
            lv2: R.Shape(ndim=2) = R.tensor_to_shape(target_shape)
            gv: R.Tensor(dtype="float32", ndim=2) = R.reshape(data, lv2)
            R.output(gv)
        return gv
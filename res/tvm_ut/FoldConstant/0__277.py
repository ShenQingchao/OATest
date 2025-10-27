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
        \"value\": \"1\"
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
    }
  ], 
  \"b64ndarrays\": [
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAABAAQABAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAA=\"
  ], 
  \"attrs\": {\"tvm_version\": \"0.17.dev0\"}
}""")
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((5, 4, 3, 2), dtype="float32")) -> R.Tensor((1, 1), dtype="int64"):
        with R.dataflow():
            lv: R.Tensor((4,), dtype="int64") = R.shape_to_tensor(R.shape([5, 4, 3, 2]))
            lv1: R.Tensor((1,), dtype="int64") = R.take(lv, metadata["relax.expr.Constant"][0], axis=0)
            lv2: R.Tensor((1, 1), dtype="int64") = R.expand_dims(lv1, axis=[0])
            gv: R.Tensor((1, 1), dtype="int64") = R.concat((lv2,), axis=0)
            R.output(gv)
        return gv
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
        \"dtype\": \"int32\", 
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
        \"value\": \"3\"
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
        \"dtype\": \"int32\", 
        \"ndim\": \"1\", 
        \"span\": \"0\"
      }
    }
  ], 
  \"b64ndarrays\": [
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAAgAQADAAAAAAAAAAwAAAAAAAAAAQAAAAIAAAADAAAA\"
  ], 
  \"attrs\": {\"tvm_version\": \"0.17.dev0\"}
}""")
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def prim_values() -> R.Prim(value=3):
        x: R.Prim(value=1) = R.prim_value(1)
        y: R.Prim(value=2) = R.prim_value(2)
        z: R.Prim(value=3) = R.prim_value(3)
        return z

    @R.function
    def shapes() -> R.Shape([7, 8, 9]):
        s1: R.Shape([1, 2, 3]) = R.shape([1, 2, 3])
        s2: R.Shape([4, 5, 6]) = R.shape([4, 5, 6])
        s3: R.Shape([7, 8, 9]) = R.shape([7, 8, 9])
        return s3

    @R.function
    def tuples_and_const(x: R.Tensor, y: R.Tensor) -> R.Tensor((3,), dtype="int32"):
        t1: R.Tuple(R.Tensor, R.Tensor, R.Tensor) = x, y, x
        t2: R.Tuple(R.Tensor, R.Tensor, R.Tensor) = y, y, x
        c: R.Tensor((3,), dtype="int32") = metadata["relax.expr.Constant"][0]
        return c

    @R.function
    def main(t: R.Tuple(R.Tensor, R.Tensor)) -> R.Tensor:
        x: R.Tensor = t[0]
        y: R.Tensor = t[1]
        z: R.Tensor = R.add(x, y)
        w: R.Tensor = R.multiply(z, z)
        return w
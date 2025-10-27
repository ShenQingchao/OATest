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
        \"_checked_type_\": \"11\", 
        \"data\": \"0\", 
        \"span\": \"0\", 
        \"struct_info_\": \"4\"
      }
    }, 
    {
      \"type_key\": \"relax.TensorStructInfo\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"2\", 
        \"shape\": \"5\", 
        \"span\": \"0\", 
        \"vdevice\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.expr.ShapeExpr\", 
      \"attrs\": {
        \"_checked_type_\": \"10\", 
        \"span\": \"0\", 
        \"struct_info_\": \"9\", 
        \"values\": \"6\"
      }
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [7, 8]
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"16\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"int64\", 
        \"span\": \"0\", 
        \"value\": \"16\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeStructInfo\", 
      \"attrs\": {
        \"ndim\": \"2\", 
        \"span\": \"0\", 
        \"values\": \"6\"
      }
    }, 
    {
      \"type_key\": \"relax.ShapeType\", 
      \"attrs\": {
        \"ndim\": \"2\", 
        \"span\": \"0\"
      }
    }, 
    {
      \"type_key\": \"relax.DynTensorType\", 
      \"attrs\": {
        \"dtype\": \"float32\", 
        \"ndim\": \"2\", 
        \"span\": \"0\"
      }
    }
  ], 
  \"b64ndarrays\": [
    \"P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAgAAAAIgAQAQAAAAAAAAABAAAAAAAAAAAAQAAAAAAAAAAAAAAACAPwAAAEAAAEBAAACAQAAAoEAAAMBAAADgQAAAAEEAABBBAAAgQQAAMEEAAEBBAABQQQAAYEEAAHBBAACAQQAAiEEAAJBBAACYQQAAoEEAAKhBAACwQQAAuEEAAMBBAADIQQAA0EEAANhBAADgQQAA6EEAAPBBAAD4QQAAAEIAAARCAAAIQgAADEIAABBCAAAUQgAAGEIAABxCAAAgQgAAJEIAAChCAAAsQgAAMEIAADRCAAA4QgAAPEIAAEBCAABEQgAASEIAAExCAABQQgAAVEIAAFhCAABcQgAAYEIAAGRCAABoQgAAbEIAAHBCAAB0QgAAeEIAAHxCAACAQgAAgkIAAIRCAACGQgAAiEIAAIpCAACMQgAAjkIAAJBCAACSQgAAlEIAAJZCAACYQgAAmkIAAJxCAACeQgAAoEIAAKJCAACkQgAApkIAAKhCAACqQgAArEIAAK5CAACwQgAAskIAALRCAAC2QgAAuEIAALpCAAC8QgAAvkIAAMBCAADCQgAAxEIAAMZCAADIQgAAykIAAMxCAADOQgAA0EIAANJCAADUQgAA1kIAANhCAADaQgAA3EIAAN5CAADgQgAA4kIAAORCAADmQgAA6EIAAOpCAADsQgAA7kIAAPBCAADyQgAA9EIAAPZCAAD4QgAA+kIAAPxCAAD+QgAAAEMAAAFDAAACQwAAA0MAAARDAAAFQwAABkMAAAdDAAAIQwAACUMAAApDAAALQwAADEMAAA1DAAAOQwAAD0MAABBDAAARQwAAEkMAABNDAAAUQwAAFUMAABZDAAAXQwAAGEMAABlDAAAaQwAAG0MAABxDAAAdQwAAHkMAAB9DAAAgQwAAIUMAACJDAAAjQwAAJEMAACVDAAAmQwAAJ0MAAChDAAApQwAAKkMAACtDAAAsQwAALUMAAC5DAAAvQwAAMEMAADFDAAAyQwAAM0MAADRDAAA1QwAANkMAADdDAAA4QwAAOUMAADpDAAA7QwAAPEMAAD1DAAA+QwAAP0MAAEBDAABBQwAAQkMAAENDAABEQwAARUMAAEZDAABHQwAASEMAAElDAABKQwAAS0MAAExDAABNQwAATkMAAE9DAABQQwAAUUMAAFJDAABTQwAAVEMAAFVDAABWQwAAV0MAAFhDAABZQwAAWkMAAFtDAABcQwAAXUMAAF5DAABfQwAAYEMAAGFDAABiQwAAY0MAAGRDAABlQwAAZkMAAGdDAABoQwAAaUMAAGpDAABrQwAAbEMAAG1DAABuQwAAb0MAAHBDAABxQwAAckMAAHNDAAB0QwAAdUMAAHZDAAB3QwAAeEMAAHlDAAB6QwAAe0MAAHxDAAB9QwAAfkMAAH9D\"
  ], 
  \"attrs\": {\"tvm_version\": \"0.17.dev0\"}
}""")
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main() -> R.Tensor((16, 16), dtype="float32"):
        with R.dataflow():
            gv: R.Tensor((16, 16), dtype="float32") = R.add(metadata["relax.expr.Constant"][0], metadata["relax.expr.Constant"][0])
            R.output(gv)
        return gv
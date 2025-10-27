import random

pass_levels = {
    "LambdaLift": "dataflow",
    "ToNonDataflow": "block",
    "RemovePurityChecking": "dataflow",
    "CallTIRRewrite": "dataflow",
    "RewriteDataflowReshape": "block",
    "StaticPlanBlockMemory": "block",
    "AttachGlobalSymbol": "dataflow",
    "Normalize": "block",
    "NormalizeGlobalVar": "dataflow",
    "CanonicalizeBindings": "block",
    "EliminateCommonSubexpr": "block",
    "BindParams": "dataflow",
    "BindSymbolicVars": "dataflow",
    "FoldConstant": "block",
    "LegalizeOps": "node",
    "RealizeVDevice": "dataflow",
    "LiftTransformParams": "dataflow",
    "UpdateVDevice": "dataflow",
    "ExpandTupleArguments": "dataflow",
    "RemoveUnusedParameters": "dataflow",
    "RemoveUnusedOutputs": "dataflow",
    "AnnotateTIROpPattern": "block",
    "FuseOps": "block",
    "FuseOpsByPattern": "block",
    "MergeCompositedataflows": "dataflow",
    "FuseTIR": "dataflow",
    "RunCodegen": "dataflow",
    "DecomposeOpsForInference": "block",
    "DecomposeOpsForTraining": "block",
    "AlterOpImpl": "node",
    "ConvertLayout": "dataflow",
    "ConvertToDataflow": "block",
    "DeadCodeElimination": "dataflow",
    "DataflowUseInplaceCalls": "dataflow",
    "ToMixedPrecision": "dataflow",
    "RewriteCUDAGraph": "dataflow",
    "FewShotTuning": "dataflow",
    "InlinePrivatedataflows": "dataflow",
    "ReorderPermuteDimsAfterConcat": "block",
    "AllocateWorkspace": "dataflow",
    "Gradient": "dataflow",
    "UpdateParamStructInfo": "dataflow",
    "ExpandMatmulOfSum": "block",
    "ReorderTakeAfterMatmul": "block",
    "KillAfterLastUse": "block",
    "MetaScheduleApplyDatabase": "dataflow",
    "BundleModelParams": "dataflow",
    "VMBuiltinLower": "block",
    "AdjustMatmulOrder": "block",
    "MetaScheduleTuneIRMod": "dataflow",
    "ComputePrimValue": "dataflow",
    "MetaScheduleTuneTIR": "dataflow",
    "LowerAllocTensor": "block",
    "TopologicalSort": "block",
    "CombineParallelMatmul": "block",
    "LazySetOutput": "dataflow",
    "LazyGetInput": "dataflow",
    "VMShapeLower": "block",
    "InlinePrivateFunctions": "dataflow",
    "MergeCompositeFunctions": "dataflow",
}


def get_level(passes):
    all_pass_level = []
    for item in passes:
        if item in pass_levels.keys():
            this_pass_level = pass_levels[item]
            if this_pass_level == 'node':
                this_pass_level = 'block'
            all_pass_level.append(this_pass_level)
        else:
            assert False, f"Cannot get the the level of pass: '{item}' "
    all_pass_level = list(all_pass_level)
    current_level = random.choice(all_pass_level)
    return current_level


if __name__ == '__main__':
    res = get_level(['ConvertToDataflow', 'ConvertLayout', 'ConvertLayout'])
    dir_ut = "../res/tvm_ut"
    import os
    all_pass = os.listdir(dir_ut)
    print(all_pass)
    for _pass in all_pass:
        res = get_level([_pass])
        print(res)
    print(res)

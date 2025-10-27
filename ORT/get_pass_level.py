import random

pass_levels = {
    "BiasDropoutFusion": "block",
    "BiasGeluFusion": "block",
    "BiasSoftmaxFusion": "block",
    "CastFloat16Transformer": "dataflow",
    "CommonSubexpressionElimination": "block",
    "ConcatSliceElimination": "dataflow",
    "ConstantFolding": "block",
    "ConstantSharing": "block",
    "ConvActivationFusion": "block",
    "ConvAddActivationFusion": "block",
    "DoubleQDQPairsRemover": "dataflow",
    "DynamicQuantizeMatMulFusion": "dataflow",
    "EmbedLayerNormFusion": "dataflow",
    "EnsureUniqueDQForNodeUnit": "dataflow",
    "FastGeluFusion": "block",
    "FreeDimensionOverrideTransformer": "dataflow",
    "GatherSliceToSplitFusion": "dataflow",
    "GeluApproximation": "dataflow",
    "GeluFusionL1": "block",
    "GeluFusionL2": "block",
    "GemmActivationFusion": "block",
    "LayerNormFusionL1": "dataflow",
    "Level1_RuleBasedTransformer": "dataflow",
    "Level2_RuleBasedTransformer": "dataflow",
    "MatMulAddFusion": "block",
    "MatMulIntegerToFloatFusion": "dataflow",
    "MatMulNBitsFusion": "dataflow",
    "MatMulScaleFusion": "dataflow",
    "MatmulTransposeFusion": "block",
    "NhwcTransformer": "dataflow",
    "PropagateCastOps": "dataflow",
    "QDQFinalCleanupTransformer": "dataflow",
    "QDQPropagationTransformer": "dataflow",
    "QDQS8ToU8Transformer": "dataflow",
    "QDQSelectorActionTransformer": "dataflow",
    "QuickGeluFusion": "dataflow",
    "RemoveDuplicateCastTransformer": "block",
    "ReshapeFusion": "block",
    "RuleTransformer": "dataflow",
    "RuleTransformer1": "dataflow",
    "RuleTransformerL1": "dataflow",
    "SkipLayerNormFusion": "block",
    "TransposeOptimizer": "block",
    "FuseReluClip": "block",
    "ConvMulFusion": "block",
    "GemmTransposeFusion": "block",
    "NoopElimination": "node",
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
            all_pass_level.append("dataflow")
            print(f"[Error] Cannot get the the level of pass: '{item}' ")
            # assert False
    all_pass_level = list(all_pass_level)
    current_level = random.choice(all_pass_level)
    return current_level


if __name__ == '__main__':
    res = get_level(['ConvertToDataflow', 'ConvertLayout', 'ConvertLayout'])
    print(res)
    import os

    os.listdir()


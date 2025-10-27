import splice_block
import splice_graph


def synthesize(base_irs, donor_irs, pass_level='dataflow'):
    # synthesize_res = splice_graph.combine_models_random(base_irs, donor_irs)
    # return synthesize_res
    try:
        if pass_level == 'block':
            synthesize_res = splice_block.combine_models_random(base_irs, donor_irs)
        elif pass_level == 'dataflow':
            synthesize_res = splice_graph.combine_models_random(base_irs, donor_irs)
        else:
            assert False, f"Cannot identify the CG level {pass_level}"
    except Exception as e:
        print(e)
        return False
    return synthesize_res


if __name__ == '__main__':
    pass


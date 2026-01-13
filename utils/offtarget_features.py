def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)

def mismatch_positions(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]

def weighted_mismatch_score(a, b):
    score = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            score += (20 - i)  # PAM-proximal penalty
    return score

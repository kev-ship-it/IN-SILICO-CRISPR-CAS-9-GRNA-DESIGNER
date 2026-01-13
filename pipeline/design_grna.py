from inference.grna_generator import generate_grnas
from inference.on_target_predict import predict_on_target
from inference.off_target_predict import predict_off_target
from inference.rank_grnas import rank

def design_best_grna(dna, cas9_type):
    """
    Design the best gRNA for a given DNA sequence and Cas9 variant.
    """
    # Pass Cas9 type to the generator
    candidates = generate_grnas(dna, cas9_type=cas9_type)
    results = []

    for seq, start, end in candidates:
        target_seq = dna[start:end]
        results.append({
            "seq": seq,
            "start": start,
            "end": end,
            "on": predict_on_target(seq, target_seq),
            "off": predict_off_target(seq, target_seq)
        })

    if not results:
        return None

    ranked = rank(results)
    return ranked[0]

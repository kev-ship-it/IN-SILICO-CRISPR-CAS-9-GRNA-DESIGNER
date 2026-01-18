from inference.grna_generator import generate_grnas
from inference.on_target_predict import predict_on_target
from inference.off_target_predict import predict_off_target
from inference.rank_grnas import rank


def design_best_grna(dna, cas9_type):
    """
    Design gRNAs for a given DNA sequence and Cas9 variant.
    Returns ALL candidates + highlights the best one.
    """

    # Generate candidate gRNAs based on Cas9 PAM rules
    candidates = generate_grnas(dna, cas9_type=cas9_type)

    results = []

    for seq, start, end in candidates:
        target_seq = dna[start:end]

        on = predict_on_target(seq, target_seq)
        off = predict_off_target(seq, target_seq)

        # Combined score (you can tweak weights anytime)
        score = 0.7 * on + 0.3 * (1 - off)

        results.append({
            "seq": seq,
            "start": start,
            "end": end,
            "on": on,
            "off": off,
            "score": score
        })

    if not results:
        return None

    # Rank gRNAs (highest score first)
    ranked = rank(results)

    # Return BOTH best + all candidates
    return {
        "best": ranked[0],
        "all": ranked
    }

from utils.biology import gc_content


def rank(grnas):
    """
    Rank gRNAs using on-target, off-target, and GC balance.
    Higher score = better gRNA.
    """

    scored = []

    for g in grnas:
        gc_penalty = abs(gc_content(g["seq"]) - 0.5)

        final_score = (
            0.6 * g["on"] -
            0.3 * g["off"] -
            0.1 * gc_penalty
        )

        scored.append({
            **g,
            "score": final_score   # 🔥 STANDARDIZED KEY
        })

    return sorted(scored, key=lambda x: x["score"], reverse=True)


from utils.biology import gc_content

def rank(grnas):
    scored = []
    for g in grnas:
        score = (
            0.6 * g["on"] -
            0.3 * g["off"] -
            0.1 * abs(gc_content(g["seq"]) - 0.5)
        )
        scored.append({**g, "final": score})
    return sorted(scored, key=lambda x: x["final"], reverse=True)

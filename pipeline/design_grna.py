from inference.grna_generator import generate_grnas
from inference.on_target_predict import predict_on_target
from inference.off_target_predict import predict_off_target
from inference.rank_grnas import rank

def design_best_grna(dna, cas9_type):
    candidates = generate_grnas(dna, cas9_type=cas9_type)
    results = []

    for seq, start, end in candidates:
        # --- FIX: Extract 74bp window centered on the target ---
        # The model expects gRNA(20) + Context(54)
        # We need to grab enough flanking sequence from the DNA
        pad_dna = "N"*50 + dna + "N"*50 # Prevent index out of bounds
        adj_start = start + 50
        
        # Take 74bp starting from the gRNA position
        full_window = pad_dna[adj_start : adj_start + 74]
        
        # Split as per your Dual-Input logic:
        # gRNA (20bp) and the surrounding Context (54bp)
        g_part = full_window[0:20]
        c_part = full_window[20:74]

        # On-target uses the 74bp split
        on = predict_on_target(g_part, c_part)
        
        # Off-target uses the standard 23bp (gRNA + PAM)
        # Note: adjust this if your off-target model expects something else
        off_target_input = (seq + dna[end:end+3])[:23] 
        off = predict_off_target(seq, off_target_input)

        score = 0.7 * on + 0.3 * (1 - off)

        results.append({
            "seq": seq,
            "start": start,
            "end": end,
            "on": on,
            "off": off,
            "score": score
        })

    if not results: return None
    ranked = rank(results)
    return {"best": ranked[0], "all": ranked}
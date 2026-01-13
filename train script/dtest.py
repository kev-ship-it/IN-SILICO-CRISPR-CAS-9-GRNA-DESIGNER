def design_best_grna(dna):
    candidates = []

    for i in range(len(dna) - 23):
        seq = dna[i:i+23]

        pam = seq[20:23]  # last 3 bases

        # enforce NGG PAM
        if pam[1:] != "GG":
            continue

        candidates.append({
            "seq": seq,
            "start": i,
            "end": i + 23,
            "on": predict_on_target(seq),
            "off": predict_off_target(seq)
        })

    print(candidates)
design_best_grna("TACCGGAGATCGGATCGATACTGTACTGTCGTAGTCGATGACTGA")

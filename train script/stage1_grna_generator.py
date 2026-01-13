def generate_grnas(dna, grna_len=20):
    dna = dna.upper()
    grnas = []

    for i in range(len(dna) - grna_len - 3):
        grna = dna[i:i+grna_len]
        pam = dna[i+grna_len:i+grna_len+3]
        if pam[1:] == "GG":  # NGG

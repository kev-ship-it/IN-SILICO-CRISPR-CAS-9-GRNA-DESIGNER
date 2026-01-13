def generate_grnas(dna, cas9_type, guide_len=20):
    """
    Generate gRNA candidates for a given DNA sequence based on Cas9 type.

    Args:
        dna (str): DNA sequence (5'->3')
        guide_len (int): length of gRNA
        cas9_type (str): Cas9 variant ("SpCas9", "SaCas9", "StCas9", etc.)

    Returns:
        list of tuples: (guide_seq, start_index, end_index)
    """

    dna = dna.upper()
    grnas = []

    # PAM definitions for different Cas9 proteins
    pam_dict = {
        "SpCas9": "NGG",
        "SaCas9": "NNGRRT",
        "StCas9": "NNAGAAW",
        # add more variants if needed
    }

    pam_pattern = pam_dict.get(cas9_type.split()[0], "NGG")

    # Degenerate base mapping
    degenerate_map = {
        "A": "A",
        "C": "C",
        "G": "G",
        "T": "T",
        "N": "ACGT",
        "R": "AG",
        "Y": "CT",
        "W": "AT",
        "S": "CG",
        "M": "AC",
        "K": "GT",
        "B": "CGT",
        "D": "AGT",
        "H": "ACT",
        "V": "ACG"
    }

    # Function to check PAM match
    def pam_matches(seq, pam):
        if len(seq) < len(pam):
            return False
        for s, p in zip(seq, pam):
            if s not in degenerate_map.get(p, p):
                return False
        return True

    # Scan DNA for PAM and extract guides
    for i in range(len(dna) - guide_len - len(pam_pattern) + 1):
        guide = dna[i:i + guide_len]
        pam_seq = dna[i + guide_len:i + guide_len + len(pam_pattern)]

        if pam_matches(pam_seq, pam_pattern):
            grnas.append((guide, i, i + guide_len))

    return grnas

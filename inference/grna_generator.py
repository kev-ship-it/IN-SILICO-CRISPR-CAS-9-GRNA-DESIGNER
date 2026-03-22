# IUPAC mapping for degenerate PAMs
DEGENERATE_MAP = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "N": "ACGT", "R": "AG", "Y": "CT", "W": "AT",
    "S": "CG", "M": "AC", "K": "GT", "B": "CGT",
    "D": "AGT", "H": "ACT", "V": "ACG"
}

# Standard PAMs for Cas9 variants
PAM_DICT = {
    "SpCas9": "NGG",
    "SaCas9": "NNGRRT",
    "StCas9": "NNAGAAW",
}

def pam_matches(seq, pam_pattern):
    """
    Checks if a DNA sequence matches a degenerate PAM pattern.
    Each position is evaluated independently.
    """
    if len(seq) != len(pam_pattern):
        return False
    for s, p in zip(seq, pam_pattern):
        # s is the base in DNA, p is the character in the PAM pattern
        if s not in DEGENERATE_MAP.get(p, p):
            return False
    return True

def get_reverse_complement(seq):
    """Helper to check the antisense strand."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join(complement.get(base, base) for base in reversed(seq.upper()))

def generate_grnas(dna, cas9_type, guide_len=None):
    dna = dna.upper().strip().replace("U", "T")
    
    # Configure parameters
    if guide_len is None:
        guide_len = 21 if "SaCas9" in cas9_type else 20

    pam_pattern = "NGG"
    for key in PAM_DICT:
        if key.lower() in cas9_type.lower():
            pam_pattern = PAM_DICT[key]
            break
    
    pam_len = len(pam_pattern)
    candidates = []

    # Function to scan a single strand
    def scan_strand(sequence, is_sense=True):
        seq_len = len(sequence)
        for i in range(seq_len - guide_len - pam_len + 1):
            guide = sequence[i : i + guide_len]
            pam_seq = sequence[i + guide_len : i + guide_len + pam_len]

            if pam_matches(pam_seq, pam_pattern):
                # If antisense, calculate original coordinates
                if is_sense:
                    start, end = i, i + guide_len
                else:
                    start = len(dna) - (i + guide_len)
                    end = len(dna) - i
                
                candidates.append({
                    "seq": guide,
                    "pam": pam_seq,
                    "start": start,
                    "end": end,
                    "strand": "Sense" if is_sense else "Antisense"
                })

    # Scan both strands
    scan_strand(dna, is_sense=True)
    scan_strand(get_reverse_complement(dna), is_sense=False)
            
    return candidates

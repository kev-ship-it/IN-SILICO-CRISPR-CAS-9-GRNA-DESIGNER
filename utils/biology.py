def gc_content(seq):
    if not seq:
        return 0.0

    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)

def mismatch_count(a,b):
    return sum(x!=y for x,y in zip(a,b))

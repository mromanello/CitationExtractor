
def avg(l):
    if l is None:
        return 0.0
    return float(sum(l)) / max(len(l), 1)

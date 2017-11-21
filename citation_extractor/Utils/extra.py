
def avg(l):
    if not l:
        return 0.0
    return float(sum(l)) / max(len(l), 1)
def chunks(l, n, truncate=False):
    """Yield successive n-sized chunks from l."""
    batches = []
    for i in range(0, len(l), n):
        if truncate and len(l[i:i + n]) < n:
            continue
        batches.append(l[i:i + n])
    return batches
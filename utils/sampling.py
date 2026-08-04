def water_fill(capacities, total):
    """Distribute `total` over keys, equal shares capped at each key's capacity.

    Returns {key: allocation}. Keys that cannot meet the equal share are taken in
    full; the freed-up deficit is re-spread over the rest until the full `total` is
    placed. If sum(capacities) < total, the allocation clamps to what is available.

    Used to spread a per-party row budget over languages (preprocessing) and a
    labelling budget over stance bins (analysis), which is why it lives here rather
    than in either package.
    """
    alloc = dict.fromkeys(capacities, 0)
    active = [key for key, cap in capacities.items() if cap > 0]
    remaining = min(total, sum(capacities.values()))

    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            # Hand out the final remainder one row at a time, fullest cells first.
            for key in sorted(active, key=lambda k: capacities[k] - alloc[k], reverse=True):
                if remaining == 0:
                    break
                alloc[key] += 1
                remaining -= 1
            break
        for key in list(active):
            give = min(share, capacities[key] - alloc[key])
            alloc[key] += give
            remaining -= give
            if alloc[key] >= capacities[key]:
                active.remove(key)
    return alloc

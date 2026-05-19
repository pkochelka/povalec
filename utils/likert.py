LIKERT_MIN = 1
LIKERT_MAX = 5
LIKERT_MIDPOINT = (LIKERT_MIN + LIKERT_MAX) / 2
LIKERT_HALF_RANGE = (LIKERT_MAX - LIKERT_MIN) / 2


def likert_to_stance(likert):
    return (LIKERT_MIDPOINT - likert) / LIKERT_HALF_RANGE


def stance_to_likert(stance):
    return LIKERT_MIDPOINT - stance * LIKERT_HALF_RANGE


def flip_likert(likert):
    return LIKERT_MIN + LIKERT_MAX - likert

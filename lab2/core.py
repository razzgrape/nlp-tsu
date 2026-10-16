import warnings
from itertools import combinations
import gensim

warnings.filterwarnings("ignore", category=RuntimeWarning)

kv = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)


def noun(w):
    return f"{w}_NOUN"


def top_nouns(positive, negative=(), topn=10):
    raw = kv.most_similar(positive=list(positive), negative=list(negative), topn=topn * 10)
    return [w for w, _ in raw if w.endswith("_NOUN")][:topn]


def get_candidates(a, b, pool_size=400, n_cands=40):
    mid = (kv[a] + kv[b]) / 2
    pool = set()
    for vec in (kv[a], kv[b], mid):
        for w, _ in kv.similar_by_vector(vec, topn=pool_size):
            if w.endswith("_NOUN"):
                pool.add(w)
    pool -= {a, b}
    return sorted(pool, key=lambda w: -(kv.similarity(w, a) + kv.similarity(w, b)))[:n_cands]


def find_formula(a, b, n_cands=40):
    cands = get_candidates(a, b, n_cands=n_cands)
    best = None

    def check(pos, neg=()):
        nonlocal best
        top = top_nouns(pos, neg)
        if a in top and b in top:
            score = top.index(a) + top.index(b)
            if best is None or score < best[0]:
                best = (score, list(pos), list(neg), top)

    for pos in combinations(cands, 2):
        check(pos)

    if best is None:
        small = cands[:20]
        for pos in combinations(small, 2):
            for neg in small:
                if neg not in pos:
                    check(pos, [neg])

    return best


if __name__ == "__main__":
    A, B = noun("батарея"), noun("кирпич")

    result = find_formula(A, B)
    if result is None:
        print("Формула не найдена, увеличьте n_cands")
    else:
        _, pos, neg, top = result
        formula = " + ".join(w.split("_")[0] for w in pos)
        if neg:
            formula += " - " + " - ".join(w.split("_")[0] for w in neg)
        print("Формула:", formula, "\n")
        for i, w in enumerate(top, 1):
            mark = "  <-- заданное слово" if w in (A, B) else ""
            print(f"{i:2d}. {w.split('_')[0]:15}{mark}")
        print("\nбатарея в топ-10:", A in top)
        print("кирпич в топ-10:", B in top)
        print("форвать_NOUN" in kv.key_to_index)
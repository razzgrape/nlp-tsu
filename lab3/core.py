from transformers import pipeline, AutoTokenizer

MODEL = "DeepPavlov/rubert-base-cased"
W1, W2 = "электронная", "железнодорожная"

tok = AutoTokenizer.from_pretrained(MODEL)
for w in (W1, W2):
    print(w, "->", tok.tokenize(w))

fill = pipeline("fill-mask", model=MODEL, top_k=10)

def top10(s):
    return [(r["token_str"].strip(), round(r["score"], 4)) for r in fill(s)]

templates = [
    "Купить билет на поезд можно онлайн, ведь в России давно работает [MASK] система.",
    "Билеты на поезд продаёт единая [MASK] система.",
    "В России билеты на поезд можно купить через [MASK] систему.",
    "Для покупки билетов в России создана единая [MASK] система.",
    "Билеты на поезд продаются через единую [MASK] систему.",
]

for t in templates:
    res = top10(t)
    words = [w for w, _ in res]
    print(("OK  " if W1 in words and W2 in words else "--  "), t)
    print("    ", res)
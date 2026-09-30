import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")

morph = pymorphy3.MorphAnalyzer()


def find_pairs(text):
    pairs = []
    for sentence in sent_tokenize(text, language="russian"):
        words = [t for t in word_tokenize(sentence, language="russian") if t.isalpha()]

        for i in range(len(words) - 1):
            w1 = morph.parse(words[i])[0]
            w2 = morph.parse(words[i + 1])[0]

            if w1.tag.POS not in ("NOUN", "ADJF") or w2.tag.POS not in ("NOUN", "ADJF"):
                continue

            same_number = w1.tag.number == w2.tag.number
            same_case = w1.tag.case == w2.tag.case
            same_gender = w1.tag.number == "plur" or w1.tag.gender == w2.tag.gender

            if same_number and same_case and same_gender:
                pairs.append(w1.normal_form + " " + w2.normal_form)
    return pairs


with open("input.txt", encoding="utf-8") as f:
    text = f.read()

for pair in find_pairs(text):
    print(pair)
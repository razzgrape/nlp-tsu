import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

NAME = "ai-forever/rugpt3large_based_on_gpt2"
WORD1, WORD2 = "вспугнуть", "спиной"

# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
tok = GPT2Tokenizer.from_pretrained(NAME)
model = GPT2LMHeadModel.from_pretrained(NAME).to(device)
model.eval()


def generate(
    model, tok, text,
    do_sample=True, max_length=100, repetition_penalty=5.0,
    top_k=5, top_p=0.95, temperature=1,
    num_beams=1, no_repeat_ngram_size=3,
    num_return_sequences=1,
):
    input_ids = tok.encode(text, return_tensors="pt").to(device)
    out = model.generate(
        input_ids,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k, top_p=top_p, temperature=temperature,
        num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
        num_return_sequences=num_return_sequences,
        pad_token_id=tok.eos_token_id,
    )
    return list(map(tok.decode, out))


def check(text):
    words = re.findall(r"[а-яё]+", text.lower())
    if WORD1 in words:
        i = words.index(WORD1)
        return WORD2 in words[i + 1:]
    return False


PROMPT = "Охотник крался по болоту очень медленно, стараясь не"
SEED = 0

torch.manual_seed(SEED)
texts = generate(
    model, tok, PROMPT,
    max_length=300, repetition_penalty=1.1,
    top_k=50, top_p=0.95, temperature=1.0,
    no_repeat_ngram_size=3, num_return_sequences=5,
)

for t in texts:
    if check(t):
        print("УСПЕХ")
        print("prompt:", PROMPT)
        print("seed:", SEED)
        print(t)
        break
else:
    print("не получилось")
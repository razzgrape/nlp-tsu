import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          LogitsProcessorList, set_seed)
from zona_processor import ZonaAuthorityLogitsProcessor

MODEL = "ai-forever/rugpt3large_based_on_gpt2"
PROMPTS = [
    "Вчера я зашёл в магазин и",
    "Сегодня утром сосед рассказал мне, что",
    "Лучший рецепт борща начинается с того, что",
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).eval()

gen_kwargs = dict(max_new_tokens=70, do_sample=True, top_k=50, top_p=0.95,
                  temperature=0.9, pad_token_id=tok.eos_token_id)

for prompt in PROMPTS:
    inputs = tok(prompt, return_tensors="pt")
    plen = inputs["input_ids"].shape[1]
    print("=" * 70, "\nПРОМПТ:", prompt)

    set_seed(42)
    base = model.generate(**inputs, **gen_kwargs)
    print("\n--- без процессора ---\n", tok.decode(base[0], skip_special_tokens=True))

    proc = ZonaAuthorityLogitsProcessor(tok, prompt_len=plen)
    set_seed(42)
    out = model.generate(**inputs, **gen_kwargs,
                         logits_processor=LogitsProcessorList([proc]))
    print("\n--- Авторитет на зоне ---\n", tok.decode(out[0], skip_special_tokens=True))
    print("\nЛичное дело:", proc.dossier())
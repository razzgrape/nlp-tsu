import torch
from transformers import LogitsProcessor

FENYA = [
    "братан", "братва", "браток", "базар", "малява", "малявы", "чифир", "чифирь",
    "пахан", "пахана", "паханом", "кент", "кенты", "понятия", "понятиям",
    "общак", "общаке", "общака", "шконка", "шконке", "вертухай", "вертухаи",
    "ксива", "блатной", "блатные", "хата", "хате", "кичман", "ништяк", "фраер",
    "фраера", "беспредел", "терпила", "шмон", "нары", "нарах", "баланда",
    "смотрящий", "грев", "тюряга", "пайка", "косяк", "фене", "стрелка",
    "разборки", "наколка", "наколки", "чалиться", "кум", "ботать", "зэк", "зэки",
    "зэка", "братки", "отсидел",
]

PRISON = [
    "зона", "зоне", "зону", "срок", "срока", "сроки", "этап", "этапом", "карцер",
    "тюрьма", "тюрьму", "тюрьме", "авторитет", "камера", "камере", "камеру",
    "сидеть", "сидел", "приговор", "лагерь", "лагеря", "конвой", "побег",
    "решётка", "решётку", "заключённый", "вор", "воры", "уголовник",
    "уголовный", "тюремный",
]

POLITE = [
    "пожалуйста", "спасибо", "благодарю", "извините", "простите", "уважаемый",
    "уважаемые", "здравствуйте", "любезно", "согласно", "настоящим",
    "вышеуказанный", "приятного",
]

INTRO = ["Слышь", "Короче", "Братан", "Значит", "Ну"]

LADDER = [(3, "первоход"), (6, "мужик"), (10, "блатной"),
          (15, "смотрящий"), (10**9, "вор в законе")]


def status(rep):
    for limit, name in LADDER:
        if rep < limit:
            return name
    return LADDER[-1][1]


def encode_forms(tok, words, capitalize=True):
    forms = set()
    for w in words:
        variants = {w, w.capitalize()} if capitalize else {w}
        for v in variants:
            ids = tok.encode(" " + v, add_special_tokens=False)
            if ids:
                forms.add(tuple(ids))
    return forms


class WordTrie:
    def __init__(self, tok, slang_forms, generic_forms=()):
        self.slang = set(slang_forms)
        self.generic = set(generic_forms) - self.slang
        self.forms = self.slang | self.generic
        self.first = {} 
        for f in self.forms:
            is_slang = f in self.slang
            if len(f) == 1:
                w = 1.0 if is_slang else 0.5
            else:
                piece = tok.decode([f[0]]).strip()
                if len(piece) < (3 if is_slang else 4):
                    continue
                w = 0.9 if is_slang else 0.4
            self.first[f[0]] = max(self.first.get(f[0], 0.0), w)
        self.nxt = {}
        for f in self.forms:
            if len(f) > 1 and f[0] in self.first:
                for i in range(1, len(f)):
                    self.nxt.setdefault(f[:i], set()).add(f[i])
        self.max_len = max(len(f) for f in self.forms)

    def completed(self, tail):
        for k in range(1, min(len(tail), self.max_len) + 1):
            key = tuple(tail[-k:])
            if key in self.forms:
                return key
        return None

    def continuations(self, tail):
        out = set()
        for k in range(1, min(len(tail), self.max_len - 1) + 1):
            out |= self.nxt.get(tuple(tail[-k:]), set())
        return out


class ZonaAuthorityLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, prompt_len,
                 max_first=9.0, thirst_gain=0.4, thirst_per_rep=0.1,
                 refractory=5, initial_rep=3.0,
                 cooldown=30, cooldown_penalty=4.0,
                 polite_first_penalty=2.0, polite_completion_penalty=12.0,
                 rep_gain=1.0, rep_loss=3.0, decay=0.98,
                 intro_base=0.5, intro_gain=0.4, intro_max=4.0):
        self.prompt_len = prompt_len
        self.fenya = WordTrie(tokenizer, encode_forms(tokenizer, FENYA),
                               encode_forms(tokenizer, PRISON))
        self.polite = WordTrie(tokenizer, encode_forms(tokenizer, POLITE))
        intro_forms = encode_forms(tokenizer, INTRO, capitalize=False)
        self.intro_first = sorted({f[0] for f in intro_forms})

        self.sent_end = set()
        toks = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        for i, t in enumerate(toks):
            t = (t or "").rstrip()
            if t and t[-1] in ".!?":
                self.sent_end.add(i)

        self.cooldown = cooldown
        self.refractory = refractory
        self.initial_rep = initial_rep
        self.since = []
        self.p = dict(max_first=max_first, thirst_gain=thirst_gain,
                      thirst_per_rep=thirst_per_rep, cooldown_penalty=cooldown_penalty,
                      polite_first=polite_first_penalty,
                      polite_completion=polite_completion_penalty,
                      rep_gain=rep_gain, rep_loss=rep_loss, decay=decay,
                      intro_base=intro_base, intro_gain=intro_gain, intro_max=intro_max)
        self.rep, self.thirst, self.last_word = [], [], []
        self.history = []

    def t(self, ids, device, V):
        ids = [i for i in ids if i < V]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def __call__(self, input_ids, scores):
        B, V = scores.shape
        gen_len = input_ids.shape[1] - self.prompt_len
        p = self.p
        if gen_len <= 0 or len(self.rep) != B:
            self.rep = [self.initial_rep] * B       
            self.thirst = [0.0] * B               
            self.since = [10**6] * B             
            self.last_word = [None] * B
            self.history = [(0, status(self.initial_rep))]

        window = max(self.fenya.max_len, self.polite.max_len, self.cooldown)
        neg = torch.finfo(scores.dtype).min / 2
        for b in range(B):
            start = max(self.prompt_len, input_ids.shape[1] - window)
            tail = input_ids[b, start:].tolist()

            if gen_len > 0:
                self.rep[b] *= p["decay"]
                done = self.fenya.completed(tail)
                if done is not None:
                    if done != self.last_word[b]:
                        k = 1.0 if done in self.fenya.slang else 0.5
                        self.rep[b] += p["rep_gain"] * k
                    self.last_word[b] = done
                    self.thirst[b] = 0.0            
                    self.since[b] = 0
                else:
                    self.since[b] += 1
                    if self.since[b] > self.refractory:
                        self.thirst[b] += p["thirst_gain"] + p["thirst_per_rep"] * self.rep[b]
                if self.polite.completed(tail) is not None:
                    self.rep[b] = max(0.0, self.rep[b] - p["rep_loss"])
                if b == 0:
                    st = status(self.rep[0])
                    if st != self.history[-1][1]:
                        self.history.append((gen_len, st))
            rep = self.rep[b]

            cont = self.fenya.continuations(tail)
            if cont:
                idx = self.t(list(cont), scores.device, V)
                if idx.numel():
                    keep = scores[b, idx].clone()
                    scores[b] = neg
                    scores[b, idx] = keep
                    continue

            recent = set(tail[-self.cooldown:])
            in_pause = self.since[b] <= self.refractory
            fb = 0.0 if in_pause else min(self.thirst[b], p["max_first"])
            ids, ws, cool = [], [], []
            for i, w in self.fenya.first.items():
                if i >= V:
                    continue
                if in_pause or i in recent:
                    cool.append(i)
                else:
                    ids.append(i)
                    ws.append(w)
            if ids:
                bonus = torch.tensor(ws, dtype=scores.dtype, device=scores.device) * fb
                scores[b, torch.tensor(ids, dtype=torch.long, device=scores.device)] += bonus
            if cool:
                scores[b, torch.tensor(cool, dtype=torch.long, device=scores.device)] -= p["cooldown_penalty"]

            scores[b, self.t(list(self.polite.first), scores.device, V)] -= p["polite_first"]
            pcont = self.polite.continuations(tail)
            if pcont:
                scores[b, self.t(pcont, scores.device, V)] -= p["polite_completion"]

            if tail and tail[-1] in self.sent_end:
                ib = min(p["intro_base"] + p["intro_gain"] * rep, p["intro_max"])
                scores[b, self.t(self.intro_first, scores.device, V)] += ib
        return scores

    def dossier(self):
        return " -> ".join(f"{s} (токен {t})" for t, s in self.history)
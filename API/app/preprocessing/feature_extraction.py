import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_OK = True
except Exception:
    VADER_OK = False

try:
    import emoji as emoji_lib
    EMOJI_LIB_OK = True
except ImportError:
    EMOJI_LIB_OK = False

EMOTIONS = ["anger", "joy", "anticipation", "disgust", "sadness", "fear", "surprise"]
NEGATIONS = {
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
    "without", "hardly", "scarcely", "barely", "cant", "cannot", "wont",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "havent",
    "hasnt", "hadnt", "shouldnt", "wouldnt", "couldnt", "neednt", "aint",
    "less", "few", "little", "rarely", "seldom",
}
INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "absolutely": 2.0, "totally": 1.8,
    "utterly": 2.0, "incredibly": 1.8, "really": 1.4, "so": 1.3,
    "highly": 1.6, "deeply": 1.6, "terribly": 1.8, "awfully": 1.8,
    "insanely": 1.8, "ridiculously": 1.8, "completely": 1.7, "fully": 1.5,
    "quite": 1.2, "rather": 1.2, "pretty": 1.2, "fairly": 1.1,
    "super": 1.6, "mega": 1.7, "seriously": 1.5, "truly": 1.5,
    "genuinely": 1.4, "profoundly": 1.7, "exceptionally": 1.7,
}
TWITTER_SLANG = {
    "trash": ("anger", -1), "scam": ("anger", -1), "lied": ("anger", -1),
    "liar": ("anger", -1), "delayed": ("anger", -1), "useless": ("anger", -1),
    "pathetic": ("anger", -1), "unacceptable": ("anger", -1),
    "rip": ("sadness", -1), "ugh": ("disgust", -1), "smh": ("disgust", -1),
    "wtf": ("anger", -1), "omg": ("surprise", 0), "lol": ("joy", 1),
    "lmao": ("joy", 1), "lit": ("joy", 1), "fire": ("joy", 1),
    "broken": ("sadness", -1), "dead": ("sadness", -1), "frozen": ("sadness", -1),
    "stuck": ("sadness", -1), "worst": ("anger", -1), "sucks": ("anger", -1),
    "disappointed": ("sadness", -1), "frustrated": ("anger", -1),
    "annoyed": ("anger", -1), "disgusted": ("disgust", -1),
    "scared": ("fear", -1), "worried": ("fear", -1), "shocked": ("surprise", 0),
    "excited": ("anticipation", 1), "thrilled": ("joy", 1),
    "angry": ("anger", -1), "furious": ("anger", -1), "mad": ("anger", -1),
    "livid": ("anger", -1), "horrible": ("anger", -1), "awful": ("anger", -1),
    "terrible": ("anger", -1),
}
EMOJI_EMOTION = {
    "\U0001F620": ("anger", -1), "\U0001F621": ("anger", -1),
    "\U0001F92C": ("anger", -1), "\U0001F4A2": ("anger", -1),
    "\U0001F600": ("joy", 1), "\U0001F601": ("joy", 1),
    "\U0001F602": ("joy", 1), "\U0001F603": ("joy", 1),
    "\U0001F604": ("joy", 1), "\U0001F605": ("joy", 1),
    "\U0001F606": ("joy", 1), "\U0001F60A": ("joy", 1),
    "\U0001F60D": ("joy", 1), "\U0001F618": ("joy", 1),
    "\U0001F970": ("joy", 1), "\U0001F929": ("joy", 1),
    "\U0001F973": ("joy", 1), "\u2764": ("joy", 1),
    "\U0001F496": ("joy", 1), "\U0001F44D": ("joy", 1),
    "\U0001F389": ("joy", 1), "\U0001F923": ("joy", 1),
    "\U0001F614": ("sadness", -1), "\U0001F622": ("sadness", -1),
    "\U0001F62D": ("sadness", -1), "\U0001F625": ("sadness", -1),
    "\U0001F641": ("sadness", -1), "\U0001F494": ("sadness", -1),
    "\U0001F979": ("sadness", -1), "\U0001F628": ("fear", -1),
    "\U0001F630": ("fear", -1), "\U0001F631": ("fear", -1),
    "\U0001F62E": ("surprise", 0), "\U0001F632": ("surprise", 0),
    "\U0001F633": ("surprise", 0), "\U0001F92F": ("surprise", 0),
    "\U0001F922": ("disgust", -1), "\U0001F910": ("disgust", -1),
    "\U0001F91E": ("anticipation", 1), "\U0001F440": ("anticipation", 0),
    "\U0001F914": ("anticipation", 0), "\U0001F44E": ("anger", -1),
}
SCORE_COEFS = {
    "anger":        ( 0.286375,  0.106408,  0.056587,  0.077908,  0.314258),
    "anticipation": (-0.054152, -0.043740, -0.054168,  0.040895, -0.036298),
    "disgust":      (-0.006201,  0.001100, -0.008199, -0.000127,  0.031900),
    "fear":         ( 0.112380,  0.004505,  0.009538,  0.011695,  0.251769),
    "joy":          ( 0.184977,  0.105246, -0.054050,  0.222935,  0.090882),
    "sadness":      ( 0.114282,  0.090758,  0.012956,  0.134229,  0.109073),
    "surprise":     ( 0.013429,  0.023348, -0.003475, -0.001475,  0.157419),
}

def load_lexicons(lexicon_dir: Path) -> Optional[dict]:
    lexicons = {}
    for em in EMOTIONS:
        path = lexicon_dir / f"{em}-NRC-Emotion-Lexicon.txt"
        if not path.exists():
            return None
        words = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[1] == "1":
                words.add(parts[0].lower())
            elif len(parts) == 1 and parts[0]:
                words.add(parts[0].lower())
        lexicons[em] = words
    return lexicons

def extract(text: str, lexicons: Optional[dict], negation_window: int = 3) -> dict:
    lex = lexicons or {em: set() for em in EMOTIONS}
    r = {}

    clean = re.sub(r"http\S+|www\.\S+|@\w+|#\w+|&\w+;", "", text)
    toks = re.findall(r"\b[a-zA-Z]+\b", clean.lower())
    raw = re.findall(r"\b[a-zA-Z]+\b", clean)

    r["word_count"] = len(toks)

    # NRC
    for em in EMOTIONS:
        r[f"{em}_count"] = sum(1 for t in toks if t in lex[em])

    # Negaciones
    neg_idx = [i for i, t in enumerate(toks) if t in NEGATIONS]
    r["negation_count"] = len(neg_idx)
    neg_per_em = defaultdict(int)
    for ni in neg_idx:
        for j in range(ni + 1, min(ni + 1 + negation_window, len(toks))):
            for em in EMOTIONS:
                if toks[j] in lex[em]:
                    neg_per_em[em] += 1
    r["negated_nrc_count"] = sum(neg_per_em.values())
    for em in EMOTIONS:
        r[f"negated_{em}_count"] = neg_per_em[em]

    # Intensificadores
    intens, weighted = [], defaultdict(float)
    for i, tok in enumerate(toks):
        if tok in INTENSIFIERS:
            m = INTENSIFIERS[tok]
            intens.append(m)
            for j in range(i + 1, min(i + 3, len(toks))):
                for em in EMOTIONS:
                    if toks[j] in lex[em]:
                        weighted[em] += m
    r["intensifier_count"]     = len(intens)
    r["intensifier_max"]       = round(max(intens), 2) if intens else 0.0
    r["intensified_nrc_count"] = sum(int(weighted[em]) for em in EMOTIONS)
    for em in EMOTIONS:
        r[f"{em}_weighted"] = round(weighted[em], 3)

    # Tipografía
    punct = re.findall(r"[!?.,;:\-]", clean)
    chars = re.findall(r"\S", clean)
    r["exclamation_count"] = len(re.findall(r"!", clean))
    r["multi_exclamation"]  = 1 if re.search(r"!{2,}", clean) else 0
    r["question_count"]     = len(re.findall(r"\?", clean))
    caps = [t for t in raw if t.isupper() and len(t) > 1]
    r["caps_word_count"]    = len(caps)
    r["caps_ratio"]         = round(len(caps) / max(len(toks), 1), 4)
    r["repeated_chars"]     = len(re.findall(r"([a-zA-Z])\1{2,}", clean))
    r["punct_ratio"]        = round(len(punct) / max(len(chars), 1), 4)

    # Emojis
    emoji_em, emoji_n = defaultdict(int), 0
    if EMOJI_LIB_OK:
        for item in emoji_lib.emoji_list(text):
            em, _ = EMOJI_EMOTION.get(item["emoji"], ("unknown", 0))
            if em != "unknown":
                emoji_em[em] += 1
            emoji_n += 1
    else:
        for char, (em, _) in EMOJI_EMOTION.items():
            c = text.count(char)
            emoji_em[em] += c
            emoji_n      += c
    r["emoji_count"] = emoji_n
    for em in EMOTIONS:
        r[f"emoji_{em}_count"] = emoji_em[em]

    # Slang
    slang_em = defaultdict(int)
    slang_n = slang_neg = slang_pos = 0
    for tok in toks:
        if tok in TWITTER_SLANG:
            em, pol = TWITTER_SLANG[tok]
            slang_em[em] += 1
            slang_n      += 1
            if pol > 0:   slang_pos += 1
            elif pol < 0: slang_neg += 1
    r["slang_count"]          = slang_n
    r["slang_negative_count"] = slang_neg
    r["slang_positive_count"] = slang_pos
    for em in EMOTIONS:
        r[f"slang_{em}_count"] = slang_em[em]

    # Scores compuestos
    for em in EMOTIONS:
        a, b, c, d, e = SCORE_COEFS[em]
        r[f"score_{em}"] = round(max(
            a * r[f"{em}_count"]         +
            b * r[f"{em}_weighted"]      +
            c * r[f"negated_{em}_count"] +
            d * r[f"emoji_{em}_count"]   +
            e * r[f"slang_{em}_count"],
            0.0
        ), 6)

    # VADER
    if VADER_OK:
        v = _vader.polarity_scores(text)
        r["vader_compound"] = round(v["compound"], 4)
        r["vader_positive"] = round(v["pos"],      4)
        r["vader_negative"] = round(v["neg"],      4)
        r["vader_neutral"]  = round(v["neu"],      4)
        r["vader_label"]    = (
            "POSITIVE" if v["compound"] >= 0.05 else
            "NEGATIVE" if v["compound"] <= -0.05 else "NEUTRAL"
        )
    else:
        r["vader_compound"] = r["vader_positive"] = None
        r["vader_negative"] = r["vader_neutral"]  = None
        r["vader_label"]    = None

    return r
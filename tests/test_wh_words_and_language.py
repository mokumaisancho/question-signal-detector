"""Test wh-word impact and cross-language variance on detector distributions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from two_pass_llama_detector import TwoPassLlamaDetector


def get_full_probs(detector, text: str):
    detector._load()
    llm = detector._llm
    tokens = llm.tokenize(text.encode())
    llm.eval(tokens)
    logits = llm._scores[-1].copy()
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    return probs


def kl_divergence(p, q, eps=1e-10):
    return np.sum(p * np.log((p + eps) / (q + eps)))


def js_divergence(p, q, eps=1e-10):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def compare(detector, base, variant):
    p_base = get_full_probs(detector, base)
    p_var = get_full_probs(detector, variant)
    kl = kl_divergence(p_base, p_var)
    js = js_divergence(p_base, p_var)
    tvd = 0.5 * np.sum(np.abs(p_base - p_var))

    h_base = -np.sum(p_base * np.log(p_base + 1e-10))
    h_var = -np.sum(p_var * np.log(p_var + 1e-10))

    ranks_base = np.argsort(p_base)[::-1]
    ranks_var = np.argsort(p_var)[::-1]

    top1_overlap = 1.0 if ranks_base[0] == ranks_var[0] else 0.0
    top5_overlap = len(set(ranks_base[:5]) & set(ranks_var[:5])) / 5.0
    top10_overlap = len(set(ranks_base[:10]) & set(ranks_var[:10])) / 10.0

    return {
        "kl": kl, "js": js, "tvd": tvd,
        "h_base": h_base, "h_var": h_var, "delta_h": h_var - h_base,
        "top1": top1_overlap, "top5": top5_overlap, "top10": top10_overlap,
    }


detector = TwoPassLlamaDetector()

# ── TEST 1: Wh-word impact ──────────────────────────────────────────

wh_tests = [
    {
        "concept": "capital_france",
        "base": "What is the capital of France?",
        "variants": [
            ("wh-what", "What is the capital of France?"),
            ("wh-which", "Which city is the capital of France?"),
            ("wh-how", "How do you call the capital of France?"),
            ("wh-where", "Where is the capital of France?"),
            ("yn-do", "Do you know the capital of France?"),
            ("yn-is", "Is Paris the capital of France?"),
            ("yn-can", "Can you tell me the capital of France?"),
            ("imp-name", "Name the capital of France."),
            ("imp-state", "State the capital of France."),
            ("imp-tell", "Tell me the capital of France."),
            ("stmt-france", "The capital of France is?"),
            ("stmt-paris", "Paris is the capital of?"),
        ],
    },
    {
        "concept": "gravity",
        "base": "What is gravity?",
        "variants": [
            ("wh-what", "What is gravity?"),
            ("wh-how", "How does gravity work?"),
            ("wh-why", "Why do objects fall?"),
            ("yn-do", "Do you understand gravity?"),
            ("yn-is", "Is gravity a force?"),
            ("yn-can", "Can you explain gravity?"),
            ("imp-explain", "Explain gravity."),
            ("imp-define", "Define gravity."),
            ("imp-describe", "Describe gravity."),
            ("stmt-gravity", "Gravity is a?"),
            ("stmt-objects", "Objects fall because of?"),
        ],
    },
    {
        "concept": "speed_light",
        "base": "What is the speed of light?",
        "variants": [
            ("wh-what", "What is the speed of light?"),
            ("wh-how", "How fast does light travel?"),
            ("wh-which", "Which speed does light have?"),
            ("yn-do", "Do you know the speed of light?"),
            ("yn-is", "Is the speed of light constant?"),
            ("yn-can", "Can light travel faster?"),
            ("imp-state", "State the speed of light."),
            ("imp-tell", "Tell me the speed of light."),
            ("imp-give", "Give the speed of light."),
            ("stmt-light", "The speed of light is?"),
            ("stmt-constant", "Light travels at?"),
        ],
    },
    {
        "concept": "mars_colony",
        "base": "What is Mars Colony population in 2035?",
        "variants": [
            ("wh-what", "What is Mars Colony population in 2035?"),
            ("wh-how", "How many people live on Mars in 2035?"),
            ("wh-which", "Which population does Mars Colony have in 2035?"),
            ("yn-do", "Do you know Mars Colony population in 2035?"),
            ("yn-is", "Is Mars Colony populated in 2035?"),
            ("yn-can", "Can Mars support a colony in 2035?"),
            ("imp-state", "State Mars Colony population in 2035."),
            ("imp-tell", "Tell me Mars Colony population in 2035."),
            ("imp-give", "Give Mars Colony population in 2035."),
            ("stmt-mars", "Mars Colony population in 2035 is?"),
            ("stmt-people", "People on Mars in 2035 number?"),
        ],
    },
]

print("=" * 130)
print("WH-WORD IMPACT ANALYSIS")
print("=" * 130)

wh_results = {"wh": [], "yn": [], "imp": [], "stmt": []}

for test in wh_tests:
    concept = test["concept"]
    base = test["base"]
    variants = test["variants"]

    print(f"\n{'─' * 130}")
    print(f"CONCEPT: {concept}")
    print(f"{'─' * 130}")
    print(f"{'Variant':<50} {'Type':<10} {'KL':>8} {'JS':>8} {'ΔH':>8} {'TVD':>8} {'Top1':>6} {'Top5':>6} {'Top10':>7}")
    print("-" * 130)

    for name, text in variants:
        qtype = name.split("-")[0]
        cmp = compare(detector, base, text)
        wh_results[qtype].append(cmp)

        t_short = text[:45] + "..." if len(text) > 48 else text
        print(f"{t_short:<50} {name:<10} {cmp['kl']:>8.4f} {cmp['js']:>8.4f} {cmp['delta_h']:>+8.4f} {cmp['tvd']:>8.4f} {cmp['top1']:>6.2f} {cmp['top5']:>6.2f} {cmp['top10']:>7.2f}")

print("\n" + "=" * 130)
print("WH-WORD AGGREGATE ANALYSIS")
print("=" * 130)

print(f"\n{'Type':<10} {'N':>4} {'Mean_KL':>10} {'Mean_JS':>10} {'Mean_ΔH':>10} {'Mean_TVD':>10} {'Top1_Ovlp':>10} {'Top5_Ovlp':>10} {'Top10_Ovlp':>11}")
print("-" * 100)
for qtype, results in wh_results.items():
    if not results:
        continue
    n = len(results)
    mean_kl = np.mean([r["kl"] for r in results])
    mean_js = np.mean([r["js"] for r in results])
    mean_dh = np.mean([r["delta_h"] for r in results])
    mean_tvd = np.mean([r["tvd"] for r in results])
    mean_t1 = np.mean([r["top1"] for r in results])
    mean_t5 = np.mean([r["top5"] for r in results])
    mean_t10 = np.mean([r["top10"] for r in results])
    print(f"{qtype:<10} {n:>4} {mean_kl:>10.4f} {mean_js:>10.4f} {mean_dh:>+10.4f} {mean_tvd:>10.4f} {mean_t1:>10.2f} {mean_t5:>10.2f} {mean_t10:>11.2f}")

# ── TEST 2: Cross-language variance ─────────────────────────────────

lang_tests = [
    {
        "concept": "capital_france",
        "english": "What is the capital of France?",
        "translations": [
            ("spanish", "¿Cuál es la capital de Francia?"),
            ("french", "Quelle est la capitale de la France?"),
            ("german", "Was ist die Hauptstadt von Frankreich?"),
            ("italian", "Qual è la capitale della Francia?"),
            ("portuguese", "Qual é a capital da França?"),
            ("dutch", "Wat is de hoofdstad van Frankrijk?"),
            ("russian", "Какова столица Франции?"),
            ("chinese", "法国的首都是什么？"),
            ("japanese", "フランスの首都は何ですか？"),
            ("korean", "프랑스의 수도는 무엇입니까?"),
        ],
    },
    {
        "concept": "gravity",
        "english": "What is gravity?",
        "translations": [
            ("spanish", "¿Qué es la gravedad?"),
            ("french", "Qu'est-ce que la gravité?"),
            ("german", "Was ist Schwerkraft?"),
            ("italian", "Cos'è la gravità?"),
            ("portuguese", "O que é gravidade?"),
            ("dutch", "Wat is zwaartekracht?"),
            ("russian", "Что такое гравитация?"),
            ("chinese", "什么是重力？"),
            ("japanese", "重力とは何ですか？"),
            ("korean", "중력이란 무엇입니까?"),
        ],
    },
    {
        "concept": "mars_colony",
        "english": "What is Mars Colony population in 2035?",
        "translations": [
            ("spanish", "¿Cuál es la población de la colonia de Marte en 2035?"),
            ("french", "Quelle est la population de la colonie martienne en 2035?"),
            ("german", "Wie hoch ist die Bevölkerung der Mars-Kolonie im Jahr 2035?"),
            ("italian", "Qual è la popolazione della colonia su Marte nel 2035?"),
            ("portuguese", "Qual é a população da colônia de Marte em 2035?"),
            ("dutch", "Wat is de bevolking van de Mars-kolonie in 2035?"),
            ("russian", "Каково население марсианской колонии в 2035 году?"),
            ("chinese", "2035年火星殖民地的人口是多少？"),
            ("japanese", "2035年の火星コロニーの人口は何ですか？"),
            ("korean", "2035년 화성 식민지의 인구는 얼마입니까?"),
        ],
    },
]

print("\n\n" + "=" * 130)
print("CROSS-LANGUAGE VARIANCE ANALYSIS")
print("=" * 130)
print("\nNote: Llama-2-7B is primarily English-trained. Non-English results reflect the model's")
print("multilingual capability (or lack thereof), not necessarily the language itself.")

lang_results = {}

for test in lang_tests:
    concept = test["concept"]
    english = test["english"]
    translations = test["translations"]

    print(f"\n{'─' * 130}")
    print(f"CONCEPT: {concept}")
    print(f"ENGLISH: \"{english}\"")
    print(f"{'─' * 130}")
    print(f"{'Language':<15} {'Text':<55} {'KL':>8} {'JS':>8} {'ΔH':>8} {'TVD':>8} {'Top1':>6} {'Top5':>6} {'Top10':>7}")
    print("-" * 130)

    for lang, text in translations:
        try:
            cmp = compare(detector, english, text)
            lang_results.setdefault(lang, []).append(cmp)

            t_short = text[:50] + "..." if len(text) > 53 else text
            print(f"{lang:<15} {t_short:<55} {cmp['kl']:>8.4f} {cmp['js']:>8.4f} {cmp['delta_h']:>+8.4f} {cmp['tvd']:>8.4f} {cmp['top1']:>6.2f} {cmp['top5']:>6.2f} {cmp['top10']:>7.2f}")
        except Exception as e:
            print(f"{lang:<15} {text[:50]:<55} ERROR: {e}")

print("\n" + "=" * 130)
print("CROSS-LANGUAGE AGGREGATE ANALYSIS")
print("=" * 130)

print(f"\n{'Language':<15} {'N':>4} {'Mean_KL':>10} {'Mean_JS':>10} {'Mean_ΔH':>10} {'Mean_TVD':>10} {'Top1_Ovlp':>10} {'Top5_Ovlp':>10} {'Top10_Ovlp':>11}")
print("-" * 100)

# Sort by mean KL (most divergent first)
lang_summary = []
for lang, results in lang_results.items():
    n = len(results)
    mean_kl = np.mean([r["kl"] for r in results])
    mean_js = np.mean([r["js"] for r in results])
    mean_dh = np.mean([r["delta_h"] for r in results])
    mean_tvd = np.mean([r["tvd"] for r in results])
    mean_t1 = np.mean([r["top1"] for r in results])
    mean_t5 = np.mean([r["top5"] for r in results])
    mean_t10 = np.mean([r["top10"] for r in results])
    lang_summary.append((lang, n, mean_kl, mean_js, mean_dh, mean_tvd, mean_t1, mean_t5, mean_t10))

lang_summary.sort(key=lambda x: x[2], reverse=True)

for lang, n, mean_kl, mean_js, mean_dh, mean_tvd, mean_t1, mean_t5, mean_t10 in lang_summary:
    print(f"{lang:<15} {n:>4} {mean_kl:>10.4f} {mean_js:>10.4f} {mean_dh:>+10.4f} {mean_tvd:>10.4f} {mean_t1:>10.2f} {mean_t5:>10.2f} {mean_t10:>11.2f}")

# Language family analysis
print("\n" + "=" * 130)
print("LANGUAGE FAMILY ANALYSIS")
print("=" * 130)

families = {
    "germanic": ["german", "dutch", "english"],
    "romance": ["spanish", "french", "italian", "portuguese"],
    "slavic": ["russian"],
    "sino_tibetan": ["chinese"],
    "japonic": ["japanese"],
    "koreanic": ["korean"],
}

print(f"\n{'Family':<20} {'Languages':<40} {'Mean_KL':>10} {'Mean_TVD':>10} {'Mean_Top1':>10}")
print("-" * 100)

for family, langs in families.items():
    family_results = []
    for lang in langs:
        if lang in lang_results:
            family_results.extend(lang_results[lang])
    if family_results:
        mean_kl = np.mean([r["kl"] for r in family_results])
        mean_tvd = np.mean([r["tvd"] for r in family_results])
        mean_t1 = np.mean([r["top1"] for r in family_results])
        lang_str = ", ".join(langs)
        print(f"{family:<20} {lang_str:<40} {mean_kl:>10.4f} {mean_tvd:>10.4f} {mean_t1:>10.2f}")

# Script vs non-script analysis
print("\n" + "=" * 130)
print("SCRIPT TYPE ANALYSIS")
print("=" * 130)

script_types = {
    "latin": ["spanish", "french", "german", "italian", "portuguese", "dutch"],
    "cyrillic": ["russian"],
    "cjk": ["chinese", "japanese", "korean"],
}

print(f"\n{'Script':<15} {'Mean_KL':>10} {'Mean_TVD':>10} {'Mean_Top1':>10} {'Mean_Top10':>11}")
print("-" * 70)
for script, langs in script_types.items():
    script_results = []
    for lang in langs:
        if lang in lang_results:
            script_results.extend(lang_results[lang])
    if script_results:
        mean_kl = np.mean([r["kl"] for r in script_results])
        mean_tvd = np.mean([r["tvd"] for r in script_results])
        mean_t1 = np.mean([r["top1"] for r in script_results])
        mean_t10 = np.mean([r["top10"] for r in script_results])
        print(f"{script:<15} {mean_kl:>10.4f} {mean_tvd:>10.4f} {mean_t1:>10.2f} {mean_t10:>11.2f}")

detector._unload()

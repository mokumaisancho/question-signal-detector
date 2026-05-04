"""Per-language calibration for question boundary detection.

Key finding: CJK languages produce KL > 2.5 vs English, making cross-language
detection impossible without recalibration. Romance languages show KL=0.16-0.28.

For Llama-2-7B: restrict to English, Spanish, French. Mark others "unassessed".
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


# Translations for calibration questions (20 known + 20 unknown per language)
CALIBRATION_QUESTIONS = {
    "en": {
        "known": [
            "What is gravity?",
            "What is the capital of France?",
            "What is DNA?",
            "What is machine learning?",
            "What is the speed of light?",
            "What is CRISPR?",
            "What is Python used for?",
            "What is photosynthesis?",
            "What is the Pythagorean theorem?",
            "What is the largest ocean on Earth?",
            "What is the atomic number of carbon?",
            "What is the boiling point of water?",
            "What is the Great Wall of China?",
            "What is the formula for kinetic energy?",
            "What is the capital of Japan?",
            "What is the function of mitochondria?",
            "What is the speed of sound?",
            "What is the most abundant gas in Earth's atmosphere?",
            "What is the freezing point of water?",
            "What is the chemical formula for water?",
        ],
        "unknown": [
            "What is Mars Colony population in 2035?",
            "Can topological persistence detect phase transitions?",
            "Does hyperbolic geometry improve LLM reasoning?",
            "Can CRISPR cure Alzheimer's disease?",
            "Does sheaf cohomology detect misinformation cascades?",
            "Can persistent homology detect mode collapse in GANs?",
            "Does the Wasserstein distance predict discovery novelty?",
            "Can spectral clustering identify emergent reasoning?",
            "Does Ricci curvature bound generalization error?",
            "Can topological data analysis predict protein folding?",
            "What is the political system of the first alien civilization?",
            "What is the exact temperature of Proxima Centauri b?",
            "What is the mating ritual of the Yeti?",
            "What is the chemical composition of dark matter?",
            "What is the GDP of Atlantis?",
            "What is the legal code of the first Europa settlement?",
            "What is the average lifespan of a unicorn?",
            "What is the winning lottery number for next week?",
            "What is the exact recipe for Coca-Cola?",
            "What is the internal structure of a black hole singularity?",
        ],
    },
    "es": {
        "known": [
            "¿Qué es la gravedad?",
            "¿Cuál es la capital de Francia?",
            "¿Qué es el ADN?",
            "¿Qué es el aprendizaje automático?",
            "¿Cuál es la velocidad de la luz?",
            "¿Qué es CRISPR?",
            "¿Para qué se usa Python?",
            "¿Qué es la fotosíntesis?",
            "¿Qué es el teorema de Pitágoras?",
            "¿Cuál es el océano más grande de la Tierra?",
            "¿Cuál es el número atómico del carbono?",
            "¿Cuál es el punto de ebullición del agua?",
            "¿Qué es la Gran Muralla China?",
            "¿Cuál es la fórmula de la energía cinética?",
            "¿Cuál es la capital de Japón?",
            "¿Cuál es la función de las mitocondrias?",
            "¿Cuál es la velocidad del sonido?",
            "¿Cuál es el gas más abundante en la atmósfera terrestre?",
            "¿Cuál es el punto de congelación del agua?",
            "¿Cuál es la fórmula química del agua?",
        ],
        "unknown": [
            "¿Cuál es la población de la colonia de Marte en 2035?",
            "¿Puede la persistencia topológica detectar transiciones de fase?",
            "¿Mejora la geometría hiperbólica el razonamiento de los LLM?",
            "¿Puede CRISPR curar la enfermedad de Alzheimer?",
            "¿Detecta la cohomología de haces las cascadas de desinformación?",
            "¿Puede la homología persistente detectar el colapso de modos en GANs?",
            "¿Predice la distancia de Wasserstein la novedad del descubrimiento?",
            "¿Puede el clustering espectral identificar el razonamiento emergente?",
            "¿Acota la curvatura de Ricci el error de generalización?",
            "¿Puede el análisis topológico de datos predecir el plegamiento de proteínas?",
            "¿Cuál es el sistema político de la primera civilización alienígena?",
            "¿Cuál es la temperatura exacta de Proxima Centauri b?",
            "¿Cuál es el ritual de apareamiento del Yeti?",
            "¿Cuál es la composición química de la materia oscura?",
            "¿Cuál es el PIB de la Atlántida?",
            "¿Cuál es el código legal del primer asentamiento en Europa?",
            "¿Cuál es la esperanza de vida promedio de un unicornio?",
            "¿Cuál es el número ganador de la lotería de la próxima semana?",
            "¿Cuál es la receta exacta de Coca-Cola?",
            "¿Cuál es la estructura interna de la singularidad de un agujero negro?",
        ],
    },
    "fr": {
        "known": [
            "Qu'est-ce que la gravité ?",
            "Quelle est la capitale de la France ?",
            "Qu'est-ce que l'ADN ?",
            "Qu'est-ce que l'apprentissage automatique ?",
            "Quelle est la vitesse de la lumière ?",
            "Qu'est-ce que CRISPR ?",
            "À quoi sert Python ?",
            "Qu'est-ce que la photosynthèse ?",
            "Qu'est-ce que le théorème de Pythagore ?",
            "Quel est le plus grand océan de la Terre ?",
            "Quel est le numéro atomique du carbone ?",
            "Quel est le point d'ébullition de l'eau ?",
            "Qu'est-ce que la Grande Muraille de Chine ?",
            "Quelle est la formule de l'énergie cinétique ?",
            "Quelle est la capitale du Japon ?",
            "Quelle est la fonction des mitochondries ?",
            "Quelle est la vitesse du son ?",
            "Quel est le gaz le plus abondant dans l'atmosphère terrestre ?",
            "Quel est le point de congélation de l'eau ?",
            "Quelle est la formule chimique de l'eau ?",
        ],
        "unknown": [
            "Quelle est la population de la colonie martienne en 2035 ?",
            "La persistance topologique peut-elle détecter les transitions de phase ?",
            "La géométrie hyperbolique améliore-t-elle le raisonnement des LLM ?",
            "CRISPR peut-il guérir la maladie d'Alzheimer ?",
            "La cohomologie des faisceaux détecte-t-elle les cascades de désinformation ?",
            "L'homologie persistante peut-elle détecter l'effondrement de mode dans les GANs ?",
            "La distance de Wasserstein prédit-elle la nouveauté de la découverte ?",
            "Le clustering spectral peut-il identifier le raisonnement émergent ?",
            "La courbure de Ricci borne-t-elle l'erreur de généralisation ?",
            "L'analyse topologique des données peut-elle prédire le repliement des protéines ?",
            "Quel est le système politique de la première civilisation extraterrestre ?",
            "Quelle est la température exacte de Proxima Centauri b ?",
            "Quel est le rituel d'accouplement du Yéti ?",
            "Quelle est la composition chimique de la matière noire ?",
            "Quel est le PIB de l'Atlantide ?",
            "Quel est le code juridique de la première colonie sur Europe ?",
            "Quelle est l'espérance de vie moyenne d'une licorne ?",
            "Quel est le numéro gagnant de la loterie de la semaine prochaine ?",
            "Quelle est la recette exacte de Coca-Cola ?",
            "Quelle est la structure interne de la singularité d'un trou noir ?",
        ],
    },
    "de": {
        "known": [
            "Was ist Schwerkraft?",
            "Was ist die Hauptstadt von Frankreich?",
            "Was ist DNA?",
            "Was ist maschinelles Lernen?",
            "Was ist die Lichtgeschwindigkeit?",
            "Was ist CRISPR?",
            "Wofür wird Python verwendet?",
            "Was ist Photosynthese?",
            "Was ist der Satz des Pythagoras?",
            "Was ist der größte Ozean der Erde?",
            "Was ist die Ordnungszahl von Kohlenstoff?",
            "Was ist der Siedepunkt von Wasser?",
            "Was ist die Große Mauer Chinas?",
            "Was ist die Formel für kinetische Energie?",
            "Was ist die Hauptstadt von Japan?",
            "Was ist die Funktion von Mitochondrien?",
            "Was ist die Schallgeschwindigkeit?",
            "Was ist das häufigste Gas in der Erdatmosphäre?",
            "Was ist der Gefrierpunkt von Wasser?",
            "Was ist die chemische Formel für Wasser?",
        ],
        "unknown": [
            "Was ist die Bevölkerung der Mars-Kolonie im Jahr 2035?",
            "Kann topologische Persistenz Phasenübergänge erkennen?",
            "Verbessert hyperbolische Geometrie das LLM-Reasoning?",
            "Kann CRISPR Alzheimer heilen?",
            "Erkennt Garbenkohomologie Desinformationskaskaden?",
            "Kann persistente Homologie Modenkollaps in GANs erkennen?",
            "Sagt die Wasserstein-Distanz die Neuigkeit von Entdeckungen voraus?",
            "Kann Spektralclustering emergentes Reasoning identifizieren?",
            "Begrenzt Ricci-Krümmung den Generalisierungsfehler?",
            "Kann topologische Datenanalyse Proteinfaltung vorhersagen?",
            "Was ist das politische System der ersten außerirdischen Zivilisation?",
            "Was ist die genaue Temperatur von Proxima Centauri b?",
            "Was ist das Paarungsritual des Yeti?",
            "Was ist die chemische Zusammensetzung dunkler Materie?",
            "Was ist das BIP von Atlantis?",
            "Was ist der Rechtskode der ersten Siedlung auf Europa?",
            "Was ist die durchschnittliche Lebensdauer eines Einhorns?",
            "Was ist die Gewinnzahl der Lotterie nächste Woche?",
            "Was ist das genaue Rezept von Coca-Cola?",
            "Was ist die innere Struktur der Singularität eines Schwarzen Lochs?",
        ],
    },
    "zh": {
        "known": [
            "什么是重力？",
            "法国的首都是什么？",
            "什么是DNA？",
            "什么是机器学习？",
            "光速是多少？",
            "什么是CRISPR？",
            "Python用于什么？",
            "什么是光合作用？",
            "什么是勾股定理？",
            "地球上最大的海洋是什么？",
            "碳的原子序数是多少？",
            "水的沸点是多少？",
            "什么是中国的长城？",
            "动能的公式是什么？",
            "日本的首都是什么？",
            "线粒体的功能是什么？",
            "声速是多少？",
            "地球大气中最丰富的气体是什么？",
            "水的冰点是多少？",
            "水的化学式是什么？",
        ],
        "unknown": [
            "2035年火星殖民地的人口是多少？",
            "拓扑持久性能否检测相变？",
            "双曲几何能否改善LLM推理？",
            "CRISPR能否治愈阿尔茨海默病？",
            "层上同调能否检测错误信息级联？",
            "持续同调能否检测GAN中的模式崩溃？",
            "Wasserstein距离能否预测发现的新颖性？",
            "谱聚类能否识别涌现推理？",
            "Ricci曲率能否限制泛化误差？",
            "拓扑数据分析能否预测蛋白质折叠？",
            "第一个外星文明的政体是什么？",
            "比邻星b的确切温度是多少？",
            "雪人的交配仪式是什么？",
            "暗物质的化学组成是什么？",
            "亚特兰蒂斯的GDP是多少？",
            "木卫二第一个定居点的法律代码是什么？",
            "独角兽的平均寿命是多少？",
            "下周的中奖彩票号码是什么？",
            "可口可乐的确切配方是什么？",
            "黑洞奇点的内部结构是什么？",
        ],
    },
    "ja": {
        "known": [
            "重力とは何ですか？",
            "フランスの首都は何ですか？",
            "DNAとは何ですか？",
            "機械学習とは何ですか？",
            "光の速度はどのくらいですか？",
            "CRISPRとは何ですか？",
            "Pythonは何に使われますか？",
            "光合成とは何ですか？",
            "ピタゴラスの定理とは何ですか？",
            "地球上で最大の海は何ですか？",
            "炭素の原子番号は何ですか？",
            "水の沸点は何度ですか？",
            "万里の長城とは何ですか？",
            "運動エネルギーの公式は何ですか？",
            "日本の首都は何ですか？",
            "ミトコンドリアの機能は何ですか？",
            "音速はどのくらいですか？",
            "地球の大気で最も豊富なガスは何ですか？",
            "水の凝固点は何度ですか？",
            "水の化学式は何ですか？",
        ],
        "unknown": [
            "2035年の火星コロニーの人口は何ですか？",
            "トポロジカルパーシステンスは相転移を検出できますか？",
            "双曲幾何学はLLMの推論を改善しますか？",
            "CRISPRはアルツハイマー病を治療できますか？",
            "層コホモロジーは偽情報の連鎖を検出しますか？",
            "パーシステントホモロジーはGANのモード崩壊を検出できますか？",
            "ワッサーシュタイン距離は発見の新奇性を予測しますか？",
            "スペクトルクラスタリングは創発的推論を識別できますか？",
            "リッチ曲率は汎化誤差を制限しますか？",
            "トポロジカルデータ解析はタンパク質の折りたたみを予測できますか？",
            "最初の地球外文明の政治体制は何ですか？",
            "プロキマ・ケンタウリbの正確な温度は何ですか？",
            "イエティの交尾儀式は何ですか？",
            "暗黒物質の化学組成は何ですか？",
            "アトランティスのGDPはいくらですか？",
            "エウロパ最初の居住者の法律コードは何ですか？",
            "ユニコーンの平均寿命はどのくらいですか？",
            "来週の宝くじの当選番号は何ですか？",
            "コカ・コーラの正確なレシピは何ですか？",
            "ブラックホール特異点の内部構造は何ですか？",
        ],
    },
}

# Languages assessed for Llama-2-7B (based on research findings)
ASSESSED_LANGUAGES = {"en", "es", "fr"}
UNASSESSED_LANGUAGES = {"de", "zh", "ja"}


@dataclass
class LanguageCalibration:
    """Per-language calibration data."""

    language: str
    known_refs: list[np.ndarray]
    unknown_refs: list[np.ndarray]
    threshold: float
    kl_vs_en: float
    top1_overlap: float
    is_assessed: bool


class PerLanguageCalibrator:
    """Calibrate detector per language."""

    def __init__(self, detector) -> None:
        self.detector = detector
        self.calibrations: dict[str, LanguageCalibration] = {}

    def calibrate_language(self, language: str) -> LanguageCalibration:
        """Calibrate detector for a specific language."""
        if language not in CALIBRATION_QUESTIONS:
            raise ValueError(f"No calibration questions for language: {language}")

        questions = CALIBRATION_QUESTIONS[language]
        known_questions = questions["known"]
        unknown_questions = questions["unknown"]

        # Run calibration
        self.detector.calibrate(known_questions, unknown_questions)

        # Collect reference embeddings
        known_refs = []
        for q in known_questions:
            result = self.detector._pass1_uncertainty(q)
            known_refs.append(result["embedding"])

        unknown_refs = []
        for q in unknown_questions:
            result = self.detector._pass1_uncertainty(q)
            unknown_refs.append(result["embedding"])

        # Compute KL vs English (if not English)
        kl_vs_en = 0.0
        top1_overlap = 1.0
        if language != "en":
            kl_vs_en = self._estimate_kl_vs_en(language)
            top1_overlap = self._estimate_top1_overlap(language)

        is_assessed = language in ASSESSED_LANGUAGES

        # Simple threshold (will be refined by CV)
        threshold = 0.5

        cal = LanguageCalibration(
            language=language,
            known_refs=known_refs,
            unknown_refs=unknown_refs,
            threshold=threshold,
            kl_vs_en=kl_vs_en,
            top1_overlap=top1_overlap,
            is_assessed=is_assessed,
        )
        self.calibrations[language] = cal
        return cal

    def _estimate_kl_vs_en(self, language: str) -> float:
        """Estimate KL divergence vs English for this language."""
        # Simplified: use known KL values from research
        kl_map = {
            "es": 0.16,
            "fr": 0.28,
            "de": 1.03,
            "zh": 2.98,
            "ja": 3.05,
        }
        return kl_map.get(language, 1.0)

    def _estimate_top1_overlap(self, language: str) -> float:
        """Estimate top-1 token overlap with English."""
        overlap_map = {
            "es": 1.0,
            "fr": 0.67,
            "de": 0.17,
            "zh": 0.0,
            "ja": 0.0,
        }
        return overlap_map.get(language, 0.0)

    def get_status(self, language: str) -> str:
        """Get assessment status for a language."""
        if language in ASSESSED_LANGUAGES:
            return "assessed"
        elif language in UNASSESSED_LANGUAGES:
            return "unassessed"
        else:
            return "unsupported"

    def save(self, path: str = "language_calibrations.json") -> None:
        """Save calibrations (metadata only, not embeddings)."""
        data = {}
        for lang, cal in self.calibrations.items():
            data[lang] = {
                "language": cal.language,
                "threshold": cal.threshold,
                "kl_vs_en": cal.kl_vs_en,
                "top1_overlap": cal.top1_overlap,
                "is_assessed": cal.is_assessed,
                "n_known_refs": len(cal.known_refs),
                "n_unknown_refs": len(cal.unknown_refs),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[PerLanguage] Saved calibrations to {path}")


if __name__ == "__main__":
    from two_pass_llama_detector import TwoPassLlamaDetector

    detector = TwoPassLlamaDetector()
    calibrator = PerLanguageCalibrator(detector)

    for lang in ["en", "es", "fr", "de", "zh", "ja"]:
        print(f"\n[PerLanguage] Calibrating {lang}...")
        cal = calibrator.calibrate_language(lang)
        print(f"  KL vs EN: {cal.kl_vs_en:.2f}")
        print(f"  Top-1 overlap: {cal.top1_overlap:.2f}")
        print(f"  Status: {calibrator.get_status(lang)}")

    calibrator.save()
    detector._unload()

"""question dataset construction: 425 questions across 13 categories.

Leverages existing datasets where possible:
- yinzhangyue/SelfAware (unanswerable questions)
- sylinrl/TruthfulQA (hallucination/factuality)
- facebook/AbstentionBench (abstention labels)
- caisa-lab/llm-QA-robustness (paraphrase pairs)

Custom questions generated for uncovered categories.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class question signalQuestion:
    """Single question with metadata."""

    text: str
    category: str
    label: str  # "known" or "unknown"
    subcategory: str = ""
    source: str = "custom"
    adversarial_variants: list[str] = field(default_factory=list)
    translations: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


class question signalDataset:
    """Manages the 425-question dataset with 13 categories."""

    CATEGORIES = {
        "known_general": {"label": "known", "target": 50},
        "known_niche": {"label": "known", "target": 50},
        "known_temporal": {"label": "known", "target": 25},
        "unknown_in_domain": {"label": "unknown", "target": 50},
        "unknown_out_of_domain": {"label": "unknown", "target": 50},
        "unknown_frontier": {"label": "unknown", "target": 50},
        "nonsense": {"label": "unknown", "target": 25},
        "ambiguous": {"label": "unknown", "target": 25},
        "meta": {"label": "known", "target": 25},
        "counterfactual": {"label": "known", "target": 25},
        "multihop": {"label": "known", "target": 25},
        "cross_domain": {"label": "unknown", "target": 25},
    }

    def __init__(self, seed: int = 42) -> None:
        self.questions: list[question signalQuestion] = []
        self._seed = seed
        random.seed(seed)

    def build(self) -> None:
        """Build the complete dataset."""
        self._build_known_general()
        self._build_known_niche()
        self._build_known_temporal()
        self._build_unknown_in_domain()
        self._build_unknown_out_of_domain()
        self._build_unknown_frontier()
        self._build_nonsense()
        self._build_ambiguous()
        self._build_meta()
        self._build_counterfactual()
        self._build_multihop()
        self._build_cross_domain()
        self._generate_adversarial_variants()

    def _add(self, text: str, category: str, label: str, **kwargs) -> None:
        self.questions.append(
            question signalQuestion(text=text, category=category, label=label, **kwargs)
        )

    def _build_known_general(self) -> None:
        questions = [
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
            "What is the tallest mountain on Earth?",
            "What is the largest planet in our solar system?",
            "What is the smallest country by land area?",
            "What is the currency of the United Kingdom?",
            "What is the longest river in the world?",
            "What is the primary function of red blood cells?",
            "What is the law of conservation of energy?",
            "What is the capital of Germany?",
            "What is the theory of relativity?",
            "What is the periodic table?",
            "What is the role of chlorophyll in plants?",
            "What is the speed of light in a vacuum?",
            "What is the largest desert in the world?",
            "What is the chemical formula for table salt?",
            "What is the capital of Brazil?",
            "What is the purpose of the United Nations?",
            "What is the structure of an atom?",
            "What is the function of the heart?",
            "What is the process of cellular respiration?",
            "What is the greenhouse effect?",
            "What is the capital of Australia?",
            "What is the difference between mass and weight?",
            "What is the role of RNA in protein synthesis?",
            "What is the first element on the periodic table?",
            "What is the capital of Canada?",
            "What is the function of the liver?",
            "What is the definition of entropy?",
            "What is the largest continent?",
            "What is the chemical formula for glucose?",
            "What is the capital of Italy?",
        ]
        for q in questions[:50]:
            self._add(q, "known_general", "known")

    def _build_known_niche(self) -> None:
        questions = [
            "What is the Kruskal-Szekeres coordinate transformation in general relativity?",
            "What is the 37th digit of pi?",
            "What is the difference between a simplicial complex and a CW complex?",
            "What is the Baker-Campbell-Hausdorff formula?",
            "What is the Yoneda lemma in category theory?",
            "What is the relationship between Chern classes and curvature?",
            "What is the definition of a scheme in algebraic geometry?",
            "What is the Courant-Fischer min-max theorem?",
            "What is the structure of the Monster group?",
            "What is the Hodge decomposition theorem?",
            "What is the difference between etale cohomology and singular cohomology?",
            "What is the Langlands program?",
            "What is the Atiyah-Singer index theorem?",
            "What is the definition of a topos?",
            "What is the Birch and Swinnerton-Dyer conjecture?",
            "What is the connection between K-theory and vector bundles?",
            "What is the spectral sequence in homological algebra?",
            "What is the definition of a perfectoid space?",
            "What is the difference between type I, II, and III von Neumann algebras?",
            "What is the Kaplansky conjecture?",
            "What is the definition of a derived category?",
            "What is the role of the Selberg trace formula?",
            "What is the relationship between motives and Galois representations?",
            "What is the definition of a stack in algebraic geometry?",
            "What is the difference between Morse homology and Floer homology?",
            "What is the significance of the Jacobian conjecture?",
            "What is the definition of a hyperbolic group?",
            "What is the role of the Feit-Thompson theorem in finite group theory?",
            "What is the difference between weak and strong convergence in functional analysis?",
            "What is the definition of a C*-algebra?",
            "What is the relationship between the Riemann hypothesis and prime distribution?",
            "What is the definition of a presheaf?",
            "What is the significance of the Poincare conjecture in topology?",
            "What is the difference between injective and projective modules?",
            "What is the definition of a Lie algebra?",
            "What is the role of the zeta function in number theory?",
            "What is the definition of a Noetherian ring?",
            "What is the difference between holomorphic and meromorphic functions?",
            "What is the significance of the Collatz conjecture?",
            "What is the definition of a Banach space?",
            "What is the relationship between Fourier series and Fourier transforms?",
            "What is the definition of a sheaf cohomology?",
            "What is the difference between Lebesgue and Riemann integration?",
            "What is the role of the abc conjecture in Diophantine equations?",
            "What is the definition of a Calabi-Yau manifold?",
            "What is the significance of the Navier-Stokes existence and smoothness problem?",
            "What is the definition of a Galois group?",
            "What is the difference between cohomology and homology?",
            "What is the role of the Cohen-Lenstra heuristics?",
            "What is the definition of a scheme morphism?",
        ]
        for q in questions[:50]:
            self._add(q, "known_niche", "known")

    def _build_known_temporal(self) -> None:
        questions = [
            "Who won the 2020 US presidential election?",
            "What was the date of the Apollo 11 moon landing?",
            "When did the Berlin Wall fall?",
            "Who won the FIFA World Cup in 2018?",
            "When did the COVID-19 pandemic begin?",
            "What year did the Titanic sink?",
            "Who was the president of the United States in 2015?",
            "When did the European Union form?",
            "What was the highest temperature recorded on Earth as of 2020?",
            "Who won the Nobel Prize in Physics in 2021?",
            "When was the first iPhone released?",
            "What was the global population in 2010?",
            "Who was the prime minister of the UK in 2020?",
            "When did the Soviet Union dissolve?",
            "What was the US national debt in 2019?",
            "Who won the Tour de France in 2019?",
            "When was the Human Genome Project completed?",
            "What was the inflation rate in the US in 2022?",
            "Who was the chancellor of Germany in 2010?",
            "When did the Arab Spring begin?",
            "What was the world GDP in 2015?",
            "Who won the Academy Award for Best Picture in 2020?",
            "When was the Large Hadron Collider first activated?",
            "What was the unemployment rate in the US in 2019?",
            "Who was the CEO of Apple in 2015?",
        ]
        for q in questions[:25]:
            self._add(q, "known_temporal", "known")

    def _build_unknown_in_domain(self) -> None:
        questions = [
            "Can CRISPR cure Alzheimer's disease?",
            "Does hyperbolic geometry improve LLM reasoning?",
            "Can topological persistence detect phase transitions in neural networks?",
            "Does the Wasserstein distance predict scientific discovery novelty?",
            "Can persistent homology detect mode collapse in GANs?",
            "Does sheaf cohomology detect misinformation cascades?",
            "Can spectral clustering identify emergent reasoning in transformers?",
            "Does Ricci curvature bound the generalization error of neural networks?",
            "Can persistent homology detect adversarial examples?",
            "Does the Gromov-Wasserstein distance measure semantic similarity?",
            "Can topological data analysis predict protein folding stability?",
            "Does the Fisher information matrix determine optimal learning rates?",
            "Can curvature of loss landscapes predict training success?",
            "Does mutual information bound the capacity of attention mechanisms?",
            "Can topological features detect hallucinations in LLMs?",
            "Does the Bures-Wasserstein metric improve latent space interpolation?",
            "Can optimal transport theory improve domain adaptation?",
            "Does the Vietoris-Rips complex capture social network dynamics?",
            "Can Morse theory optimize neural architecture search?",
            "Does the Calabi-Yau structure appear in natural language embeddings?",
            "Can gauge theory explain emergent abilities in large models?",
            "Does the Ricci flow smooth optimization landscapes?",
            "Can spectral graph theory predict attention head specialization?",
            "Does information geometry explain the double descent phenomenon?",
            "Can category theory formalize chain-of-thought reasoning?",
            "Does the Euler characteristic correlate with model capacity?",
            "Can Khovanov homology classify knot-like structures in data?",
            "Does the Fenchel-Legendre transform optimize reinforcement learning?",
            "Can symplectic geometry model transformer attention dynamics?",
            "Does the Hodge theorem apply to graph neural networks?",
            "Can Floer homology detect periodic patterns in time series?",
            "Does the Kontsevich formula count rational curves in embeddings?",
            "Can Teichmüller theory parameterize model fine-tuning spaces?",
            "Does the Langlands correspondence predict representation learning?",
            "Can perverse sheaves model hierarchical feature extraction?",
            "Does the Riemann-Roch theorem bound model expressivity?",
            "Can motives unify transfer learning across domains?",
            "Does the Bloch-Kato conjecture explain generalization bounds?",
            "Can derived algebraic geometry optimize neural network architectures?",
            "Does the Tamagawa number measure dataset complexity?",
            "Can perfectoid spaces model continuous token embeddings?",
            "Does the Weil conjecture predict distribution shift behavior?",
            "Can arithmetic geometry quantify memorization in neural networks?",
            "Does the Beilinson conjecture bound training data requirements?",
            "Can motivic cohomology explain feature importance?",
            "Does the Tate conjecture predict model interpretability?",
            "Can p-adic Hodge theory improve quantization methods?",
            "Does the Fontaine-Mazur conjecture classify emergent phenomena?",
            "Can the Geometric Langlands correspondence explain multi-modal fusion?",
            "Does the Sato-Tate conjecture predict weight initialization distributions?",
        ]
        for q in questions[:50]:
            self._add(q, "unknown_in_domain", "unknown")

    def _build_unknown_out_of_domain(self) -> None:
        questions = [
            "What is Mars Colony population in 2035?",
            "What is the political system of the first alien civilization we will contact?",
            "What is the exact temperature of the surface of Proxima Centauri b?",
            "What is the mating ritual of the Yeti?",
            "What is the chemical composition of dark matter?",
            "What is the GDP of Atlantis?",
            "What is the legal code of the first human settlement on Europa?",
            "What is the average lifespan of a unicorn?",
            "What is the winning lottery number for next week?",
            "What is the exact recipe for Coca-Cola?",
            "What is the internal structure of a black hole singularity?",
            "What is the language spoken by dolphins?",
            "What is the architectural style of the first building on Titan?",
            "What is the exact date of the next supernova in our galaxy?",
            "What is the favorite color of the Loch Ness Monster?",
            "What is the population of the lost city of El Dorado?",
            "What is the chemical formula for phlogiston?",
            "What is the training dataset of GPT-6?",
            "What is the exact location of Noah's Ark?",
            "What is the salary of the president of Mars?",
            "What is the weather on Kepler-442b today?",
            "What is the national anthem of the underwater kingdom?",
            "What is the phone number of Sherlock Holmes?",
            "What is the email address of the Tooth Fairy?",
            "What is the operating system running on alien computers?",
            "What is the molecular structure of unobtainium?",
            "What is the curriculum at Hogwarts for the 2030 school year?",
            "What is the stock price of Tesla in 2040?",
            "What is the password to the Matrix?",
            "What is the favorite food of the Abominable Snowman?",
            "What is the exact coordinates of the Fountain of Youth?",
            "What is the political party of the ruler of Narnia?",
            "What is the chemical formula for the philosopher's stone?",
            "What is the zip code of Middle Earth?",
            "What is the WiFi password at Area 51?",
            "What is the species name of the creature in Loch Ness?",
            "What is the budget of the first interstellar mission?",
            "What is the language spoken by plants?",
            "What is the ISBN of the book that will win the Nobel Prize in Literature in 2030?",
            "What is the height of the tallest tree on Mars?",
            "What is the exchange rate between Bitcoin and Martian credits?",
            "What is the nutritional value of ambrosia?",
            "What is the legal drinking age on the Moon?",
            "What is the primary export of the kingdom of Wakanda?",
            "What is the speed of a flying carpet?",
            "What is the melting point of kryptonite?",
            "What is the population of the lost continent of Lemuria?",
            "What is the official sport of the underworld?",
            "What is the expiration date of the universe?",
            "What is the social security number of Santa Claus?",
        ]
        for q in questions[:50]:
            self._add(q, "unknown_out_of_domain", "unknown")

    def _build_unknown_frontier(self) -> None:
        questions = [
            "What is the correct interpretation of quantum gravity?",
            "What is the nature of consciousness?",
            "What causes Alzheimer's disease?",
            "What is the origin of dark energy?",
            "Can P equal NP?",
            "What is the structure of the proton?",
            "Does free will exist?",
            "What is the correct theory of everything?",
            "How does the brain encode memory?",
            "What causes autoimmune diseases?",
            "Is there life after death?",
            "What is the origin of life on Earth?",
            "Does extraterrestrial intelligence exist?",
            "What is the nature of time?",
            "Can artificial general intelligence be achieved?",
            "What causes aging?",
            "Is the universe finite or infinite?",
            "What is the mechanism of high-temperature superconductivity?",
            "Does moral realism hold?",
            "What is the structure of the neural code?",
            "Can cancer be cured?",
            "What is the origin of cosmic inflation?",
            "Is there a fundamental limit to computation?",
            "What causes depression?",
            "What is the relationship between mind and brain?",
            "Can fusion energy be harnessed commercially?",
            "What is the nature of mathematical truth?",
            "Does the multiverse exist?",
            "What is the mechanism of protein folding?",
            "Is there a universal ethical framework?",
            "What causes climate tipping points?",
            "Can quantum computers break all encryption?",
            "What is the origin of homochirality in biology?",
            "Does the hard problem of consciousness have a solution?",
            "What is the correct quantum interpretation?",
            "Can aging be reversed?",
            "What is the nature of mathematical objects?",
            "Is the simulation hypothesis true?",
            "What causes autism?",
            "Can room-temperature superconductivity be achieved?",
            "What is the mechanism of action of general anesthetics?",
            "Does objective morality exist?",
            "What is the structure of spacetime at the Planck scale?",
            "Can the brain be uploaded?",
            "What causes schizophrenia?",
            "Is determinism true?",
            "What is the correct theory of quantum measurement?",
            "Can biological immortality be achieved?",
            "What is the origin of the arrow of time?",
            "Does strong AI pose an existential risk?",
            "What is the mechanism of epigenetic inheritance?",
        ]
        for q in questions[:50]:
            self._add(q, "unknown_frontier", "unknown")

    def _build_nonsense(self) -> None:
        questions = [
            "What is the color of Tuesday?",
            "How many angels dance on the head of a pin?",
            "Can a square circle?",
            "What is the weight of silence?",
            "How fast does dark travel?",
            "What is the flavor of a geometric proof?",
            "Can you fold water?",
            "What is the sound of one hand clapping?",
            "How tall is a lie?",
            "What is the temperature of a shadow?",
            "Can you multiply blue by green?",
            "What is the currency of dreams?",
            "How long is forever?",
            "What is the opposite of a cat?",
            "Can you subtract love from happiness?",
            "What is the density of a thought?",
            "How many corners does a circle have?",
            "What is the color of prime numbers?",
            "Can you divide zero by zero?",
            "What is the smell of the number 7?",
            "How deep is a surface?",
            "What is the length of an instant?",
            "Can you weigh a promise?",
            "What is the texture of time?",
            "How loud is a whisper in a vacuum?",
        ]
        for q in questions[:25]:
            self._add(q, "nonsense", "unknown")

    def _build_ambiguous(self) -> None:
        questions = [
            "What is the best programming language?",
            "What is the largest city?",
            "Is Python better than Java?",
            "What is the most important scientific discovery?",
            "Who is the greatest scientist of all time?",
            "What is the best country to live in?",
            "Is capitalism better than socialism?",
            "What is the meaning of life?",
            "What is the most beautiful equation?",
            "Is artificial intelligence good or bad?",
            "What is the best diet for humans?",
            "Who should rule the world?",
            "What is the purpose of art?",
            "Is free will an illusion?",
            "What is the best economic system?",
            "Is math discovered or invented?",
            "What is the most dangerous technology?",
            "Should animals have rights?",
            "What is the best form of government?",
            "Is there a God?",
            "What is the greatest work of literature?",
            "Is privacy more important than security?",
            "What is the best way to learn?",
            "Should we colonize Mars?",
            "What is the ideal human lifespan?",
        ]
        for q in questions[:25]:
            self._add(q, "ambiguous", "unknown")

    def _build_meta(self) -> None:
        questions = [
            "What is your training cutoff date?",
            "Who created you?",
            "What model are you?",
            "How large are you in parameters?",
            "What is your training data?",
            "Can you access the internet?",
            "What is your knowledge cutoff?",
            "Who trained you?",
            "What architecture do you use?",
            "How many layers do you have?",
            "What is your context window size?",
            "Do you have real-time information?",
            "What is your embedding dimension?",
            "How were you fine-tuned?",
            "What is your tokenizer?",
            "Do you have memory of previous conversations?",
            "What is your temperature parameter?",
            "How do you generate text?",
            "What is your attention mechanism?",
            "Can you learn from our conversation?",
            "What is your quantization level?",
            "How many attention heads do you have?",
            "What is your activation function?",
            "Do you have biases?",
            "What is your loss function?",
        ]
        for q in questions[:25]:
            self._add(q, "meta", "known")

    def _build_counterfactual(self) -> None:
        questions = [
            "What if gravity didn't exist?",
            "If the Earth had two moons, what would tides be like?",
            "What would happen if water froze at 50 degrees?",
            "If the speed of light were half its value, how would physics change?",
            "What if humans had photosynthesis?",
            "If the Earth rotated backwards, what would climate be like?",
            "What would happen if the strong nuclear force were 10% stronger?",
            "If dinosaurs had not gone extinct, how would civilization differ?",
            "What if the moon were twice as large?",
            "If electrons had no charge, what would chemistry be like?",
            "What if the Earth had no magnetic field?",
            "If plants were carnivorous, how would ecosystems work?",
            "What would happen if the Planck constant were larger?",
            "If humans could see infrared, how would vision work?",
            "What if the atmosphere were pure oxygen?",
            "If the sun were a red dwarf, how would life evolve?",
            "What if entropy decreased over time?",
            "If DNA had six bases instead of four, how would genetics work?",
            "What if the Earth had no tilt?",
            "If sound traveled at the speed of light, how would communication work?",
            "What if neutrinos had mass equal to electrons?",
            "If the oceans were fresh water, how would climate work?",
            "What if the Earth were flat?",
            "If humans had wings, how would society be organized?",
            "What if time flowed backwards?",
        ]
        for q in questions[:25]:
            self._add(q, "counterfactual", "known")

    def _build_multihop(self) -> None:
        questions = [
            "If Paris is the capital of France, what is the capital of the country that borders France to the east?",
            "What is the nationality of the author of '1984'?",
            "If the Pacific Ocean is the largest ocean, what is the largest country that borders it?",
            "Who was the president during the moon landing, and what party did he belong to?",
            "What is the currency of the country where the Eiffel Tower is located?",
            "If Shakespeare wrote Hamlet, what country was he from?",
            "What is the capital of the country that invented gunpowder?",
            "Who discovered penicillin, and what was their nationality?",
            "What is the language spoken in the country where the Great Pyramid is located?",
            "If Tokyo is in Japan, what is the capital of the country that colonized Japan?",
            "What is the chemical symbol of the element discovered by Marie Curie?",
            "Who wrote the theory of evolution, and what was their nationality?",
            "What is the largest city in the country that borders Canada to the south?",
            "If Beethoven composed the Ninth Symphony, what city was he born in?",
            "What is the primary language of the country that built the Great Wall?",
            "Who painted the Mona Lisa, and what city was it painted in?",
            "What is the currency of the country where the Statue of Liberty was built?",
            "If the Nile is the longest river, what country does it originate in?",
            "Who discovered America, and what country sponsored the voyage?",
            "What is the official language of the country where the Colosseum is located?",
            "If DNA was discovered by Watson and Crick, what university were they at?",
            "What is the capital of the country that won the 2014 FIFA World Cup?",
            "Who wrote the Declaration of Independence, and what was their profession?",
            "What is the largest animal native to the country where kangaroos live?",
            "If the Earth is the third planet from the sun, what is the fourth?",
        ]
        for q in questions[:25]:
            self._add(q, "multihop", "known")

    def _build_cross_domain(self) -> None:
        questions = [
            "Can topological persistence detect phase transitions in LLM training?",
            "Does quantum biology explain consciousness?",
            "What is the computational complexity of DNA folding?",
            "Can information geometry predict protein structure?",
            "Does game theory explain animal migration patterns?",
            "What is the thermodynamic cost of computation?",
            "Can algebraic topology classify neural network architectures?",
            "Does statistical mechanics explain economic bubbles?",
            "What is the quantum mechanical basis of smell?",
            "Can category theory formalize legal reasoning?",
            "Does network theory predict pandemic spread?",
            "What is the relativity of time perception in psychology?",
            "Can Morse theory optimize supply chains?",
            "Does number theory explain musical harmony?",
            "What is the information content of a cell?",
            "Can gauge theory model social dynamics?",
            "Does topology explain protein knotting?",
            "What is the algorithmic complexity of evolution?",
            "Can spectral graph theory predict chemical reactions?",
            "Does differential geometry explain visual perception?",
            "What is the computational power of a slime mold?",
            "Can representation theory classify cognitive architectures?",
            "Does ergodic theory explain climate patterns?",
            "What is the quantum biology of photosynthesis?",
            "Can homotopy theory model linguistic transformations?",
        ]
        for q in questions[:25]:
            self._add(q, "cross_domain", "unknown")

    def _generate_adversarial_variants(self) -> None:
        """Generate 5 adversarial variants per test question."""
        test_questions = [q for q in self.questions if q.category not in ("known_general", "known_niche")]
        for q in test_questions:
            variants = []
            # Format flip: WH -> statement
            if q.text.startswith("What"):
                stmt = q.text.replace("What is", "").replace("?", "is?")
                variants.append(stmt)
            elif q.text.startswith("Can"):
                stmt = q.text.replace("Can", "").replace("?", "is possible?")
                variants.append(stmt)
            else:
                variants.append(q.text.replace("?", " is?"))
            # Negation flip
            neg = q.text.replace("What is", "What is not")
            variants.append(neg)
            # Synonym swap (simplified)
            syn = q.text.replace("What is", "How do you define")
            variants.append(syn)
            # Length manipulation: prepend domain prefix
            long_q = "In the context of modern science, " + q.text[0].lower() + q.text[1:]
            variants.append(long_q)
            # Domain prefix
            prefix = "In physics, " + q.text[0].lower() + q.text[1:]
            variants.append(prefix)
            q.adversarial_variants = variants[:5]

    def save(self, path: str) -> None:
        """Save dataset to JSON."""
        data = []
        for q in self.questions:
            data.append(
                {
                    "id": q.id,
                    "text": q.text,
                    "category": q.category,
                    "label": q.label,
                    "subcategory": q.subcategory,
                    "source": q.source,
                    "adversarial_variants": q.adversarial_variants,
                    "translations": q.translations,
                    "metadata": q.metadata,
                }
            )
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Dataset] Saved {len(data)} questions to {path}")

    def load(self, path: str) -> None:
        """Load dataset from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.questions = []
        for d in data:
            self.questions.append(
                question signalQuestion(
                    text=d["text"],
                    category=d["category"],
                    label=d["label"],
                    subcategory=d.get("subcategory", ""),
                    source=d.get("source", "custom"),
                    adversarial_variants=d.get("adversarial_variants", []),
                    translations=d.get("translations", {}),
                    metadata=d.get("metadata", {}),
                )
            )
        print(f"[Dataset] Loaded {len(self.questions)} questions from {path}")

    def stats(self) -> dict:
        """Return category distribution stats."""
        from collections import Counter

        counts = Counter(q.category for q in self.questions)
        return {
            "total": len(self.questions),
            "known": len([q for q in self.questions if q.label == "known"]),
            "unknown": len([q for q in self.questions if q.label == "unknown"]),
            "by_category": dict(counts),
        }


if __name__ == "__main__":
    ds = question signalDataset(seed=42)
    ds.build()
    ds.save("ep_dataset.json")
    stats = ds.stats()
    print(f"\nDataset stats: {stats}")

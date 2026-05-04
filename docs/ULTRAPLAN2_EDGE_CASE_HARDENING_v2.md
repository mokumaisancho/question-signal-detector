# ULTRAPLAN2: Edge Case Hardening & Hallucination Exploitation

**Project:** [Qq]uestion Signal Boundary Detection v2.1  
**Date:** 2026-05-04  
**Duration:** 48 hours (2 days)  
**Agents:** 4 (architect, detector-engineer, test-engineer, integration-lead)  
**MVP:** Counterfactual fix + Nonsense fix + Ambiguous fix  

---

## CRITICAL FINDING: The Reverse Exploitation

The detector's failure modes constitute a **novel attack vector for artificially inducing hallucinations** in any LLM with similar uncertainty calibration.

### How It Works

Given a detector that uses embedding distance as its primary signal:

1. **Take a question the model KNOWS** (e.g., "What is Newton's first law?")
2. **Rephrase into an embedding-exile format**:
   - Add "What if" prefix → "What if Newton's first law didn't exist?"
   - Use obscure terminology → "What is the inertial reference frame implication of the primo lex Newtoniana?"
   - Inject syntactic noise → "What is the first law of Newton in the context of non-standard inertial frames under Lorentzian manifold constraints?"
3. **The detector classifies it as UNKNOWN**
4. **The system abstains or routes to RAG**
5. **The user never gets the correct answer** — a hallucination by omission

### Attack Classification

| Attack Type | Mechanism | Detector Response | Result |
|-------------|-----------|-------------------|--------|
| **Embedding Exile** | Rephrase into unfamiliar embedding region | UNKNOWN | False abstention |
| **Syntactic Camouflage** | Nonsense with familiar grammar | KNOWN | False confidence |
| **Subjective Injection** | Opinion framed as fact | KNOWN | False confidence |
| **Meta-Confusion** | Ask about model internals inconsistently | MIXED | Unreliable |

### Why This Matters

This is not just a detector bug — it's a **systematic vulnerability**:
- Works on ANY model using embedding-distance uncertainty detection
- Requires no model access (just prompt engineering)
- Can force models to withhold correct answers OR confidently output nonsense
- Has implications for safety: adversaries can induce hallucinations by prompt design

**The fix is therefore dual-use:** hardening the detector against edge cases also closes a hallucination exploitation vector.

---

## GitHub Repos to Adopt/Reference

### Adopt (integrate or port)

| Repo | Stars | What | How We Use It |
|------|-------|------|---------------|
| [potsawee/selfcheckgpt](https://github.com/potsawee/selfcheckgpt) | 610 | Self-consistency hallucination detection | Port `selfcheckgpt.py` scoring logic for our consistency check |
| [Aishna26/hallucination_detection_in_LLMs](https://github.com/Aishna26/hallucination_detection_in_LLMs) | ~50 | Self-consistency + DBSCAN entropy | Reference for multi-sample consistency scoring |
| [codeagrawal07/LLM-Powered-Question-Classifier](https://github.com/codeagrawal07/LLM-Powered-Question-Classifier) | ~20 | LLM-based question type classification | Reference for subjective/factual/counterfactual classifier |

### Reference (learn from, don't integrate)

| Repo | What | Why Reference Only |
|------|------|-------------------|
| [Xianjun-Yang/Awesome_papers_on_LLMs_detection](https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection) | Survey of detection methods | Too broad; we need specific implementations |
| SECA (ICLR 2024) | Semantic attacks for hallucination elicitation | Shows the attack vector we're defending against |
| Drivel-ology benchmark | Nonsense detection benchmark | We implement our own coherence check |

---

## DEPENDENCY GRAPH

```
Wave 1 (Foundation)
├── W1-T1: Length-normalized scoring
│   └── BLOCKS: W2-T1, W2-T2, W2-T3, W2-T4
├── W1-T2: Semantic coherence probe (nonsense detector)
│   └── BLOCKS: W2-T3
├── W1-T3: Question-type classifier
│   └── BLOCKS: W2-T1, W2-T2, W2-T3, W2-T4
└── W1-T4: Self-consistency check (SelfCheckGPT port)
    └── BLOCKS: W2-T1, W2-T2

Wave 2 (Integration)
├── W2-T1: Counterfactual path (domain knowledge check)
│   └── DEPENDS: W1-T3, W1-T4
├── W2-T2: Subjective path (uncertainty flag)
│   └── DEPENDS: W1-T3
├── W2-T3: Nonsense path (coherence + entropy)
│   └── DEPENDS: W1-T2, W1-T3
├── W2-T4: Niche path (cluster calibration)
│   └── DEPENDS: W1-T1
└── W2-T5: Meta path (dedicated calibration)
    └── DEPENDS: W1-T3

Wave 3 (Validation)
├── W3-T1: Edge case re-run
│   └── DEPENDS: W2-T1, W2-T2, W2-T3, W2-T4, W2-T5
├── W3-T2: Hallucination attack validation
│   └── DEPENDS: W2-T1, W2-T2, W2-T3
└── W3-T3: Report generation
    └── DEPENDS: W3-T1, W3-T2
```

### Contradiction Check

| Potential Conflict | Resolution |
|-------------------|------------|
| W2-T4 (niche) vs W2-T1 (counterfactual) both modify detector.detect() | Sequential: W2-T1 adds routing, W2-T4 adds calibration. No code collision. |
| W1-T4 (self-consistency) needs model loaded for 3 samples | Resource: unload/reload handled by session context manager. No conflict. |
| W1-T2 (coherence probe) needs sentence-transformers | Dependency: install in W0 (setup). No runtime conflict. |
| Multiple agents editing same file | Rule: detector.py changes go through integration-lead. Other agents write new modules. |

---

## MVP TIERS

### MVP-0: Setup (4 hours, all agents)
- Install dependencies (sentence-transformers)
- Create branch `edge-case-hardening`
- Verify no regression on existing 100% CV pass

### MVP-1: Critical Fixes (16 hours, 3 agents)
- W1-T1: Length normalization
- W1-T3: Question-type classifier
- W2-T1: Counterfactual domain check
- W2-T2: Subjective flag
- W2-T3: Nonsense coherence check
- **Target:** Counterfactual 0% → 70%, Nonsense 48% → 70%, Ambiguous 24% → 50%

### MVP-2: Robustness (16 hours, 2 agents)
- W1-T4: Self-consistency check
- W2-T4: Niche cluster calibration
- W2-T5: Meta calibration
- W3-T1: Full edge case re-run
- **Target:** All edge cases ≥ 60%, no category < 50%

### MVP-3: Validation (12 hours, 1 agent)
- W3-T2: Hallucination attack validation
- W3-T3: Report generation
- Documentation
- **Target:** Document attack vectors, demonstrate fixes close them

---

## AGENT ASSIGNMENTS

| Agent | Role | Tasks | Skills Needed |
|-------|------|-------|---------------|
| **architect** | Design & review | W0 setup, dependency graph, contradiction checks, final review | System design, code review |
| **detector-engineer** | Core detector mods | W1-T1, W1-T3, W2-T1, W2-T2, W2-T4, W2-T5 | Python, numpy, embedding math |
| **test-engineer** | Tests & validation | W1-T2, W1-T4, W3-T1, W3-T2, W3-T3 | pytest, statistical analysis |
| **integration-lead** | Merge & orchestration | All merge conflicts, CI, regression checks | git, CI/CD, integration testing |

### Communication Protocol
- Daily standup via task list updates
- Merge requests go through integration-lead
- Blocking issues escalate to architect
- End-of-wave demos

---

## TASK DETAILS

### WAVE 0: Setup

#### W0-T1: Environment Setup
- **Assigned:** architect
- **Duration:** 2h
- **Dependencies:** None
- **Description:** Install sentence-transformers, verify model path, create branch
- **Acceptance:** `python -c "import sentence_transformers; print('OK')"` succeeds, branch exists

#### W0-T2: Baseline Regression Check
- **Assigned:** integration-lead
- **Duration:** 2h
- **Dependencies:** W0-T1
- **Description:** Run existing validation, confirm 100% CV still passes
- **Acceptance:** `python ep_validation.py` shows same results as 2026-05-04 baseline

---

### WAVE 1: Foundation

#### W1-T1: Length-Normalized Scoring
- **Assigned:** detector-engineer
- **Duration:** 4h
- **Dependencies:** W0-T2
- **Description:** Normalize hidden_norm by token count in _pass1_uncertainty()
- **Changes:**
  ```python
  # OLD
  norm_signal = (p1["hidden_norm"] - 20.0) / 10.0
  
  # NEW
  n_tokens = len(p1.get("tokens", [1]))  # or from tokenizer
  norm_per_token = p1["hidden_norm"] / max(n_tokens, 1)
  norm_signal = (norm_per_token - 2.0) / 1.0
  ```
- **Acceptance:** test_length edge case shows |correlation| < 0.3

#### W1-T2: Semantic Coherence Probe
- **Assigned:** test-engineer
- **Duration:** 6h
- **Dependencies:** W0-T2
- **Description:** Add semantic coherence check using sentence-transformers to detect nonsense
- **Changes:** New module `ep_coherence.py`
- **Approach:**
  - Encode question with sentence-transformer
  - Compare to distribution of 1000 natural questions
  - Low similarity = nonsense
- **Acceptance:** test_nonsense shows unknown_rate ≥ 70% (up from 48%)

#### W1-T3: Question-Type Classifier
- **Assigned:** detector-engineer
- **Duration:** 6h
- **Dependencies:** W0-T2
- **Description:** Rule-based classifier for counterfactual/subjective/meta/nonsense/factual
- **Changes:** New module `ep_question_type.py`
- **Rules:**
  ```python
  def classify(question: str) -> str:
      q_lower = question.lower()
      if q_lower.startswith(("what if", "if ")):
          return "counterfactual"
      if any(w in q_lower for w in ["best", "greatest", "better than", "worst", "most beautiful"]):
          return "subjective"
      if any(w in q_lower for w in ["your training", "your model", "your parameters", "your architecture"]):
          return "meta"
      return "factual"
  ```
- **Acceptance:** Unit tests for 50 questions, accuracy ≥ 90%

#### W1-T4: Self-Consistency Check (SelfCheckGPT Port)
- **Assigned:** test-engineer
- **Duration:** 8h
- **Dependencies:** W0-T2
- **Description:** Port SelfCheckGPT logic: generate answer 3x, measure consistency
- **Changes:** New module `ep_consistency.py`
- **Approach:**
  - Generate 3 answers at T=0.3, T=0.7, T=1.0
  - Encode answers with sentence-transformer
  - Compute pairwise cosine similarity
  - Low similarity = uncertain = unknown
- **Acceptance:** Counterfactual questions show consistency ≥ 0.7 (model gives same physics answer each time)

---

### WAVE 2: Integration

#### W2-T1: Counterfactual Domain Knowledge Path
- **Assigned:** detector-engineer
- **Duration:** 6h
- **Dependencies:** W1-T3, W1-T4
- **Description:** When question-type = counterfactual, use self-consistency instead of embedding distance
- **Changes:** `two_pass_llama_detector.py` detect() method
- **Logic:**
  ```python
  if question_type == "counterfactual":
      # Check if model can reason consistently about the domain
      consistency = self_consistency_check(question)
      if consistency > 0.6:
          return KNOWN  # Model knows the domain
      else:
          return UNKNOWN
  ```
- **Acceptance:** test_counterfactual accuracy ≥ 70% (up from 0%)

#### W2-T2: Subjective Uncertainty Path
- **Assigned:** detector-engineer
- **Duration:** 4h
- **Dependencies:** W1-T3
- **Description:** When question-type = subjective, flag as uncertain regardless of embedding
- **Changes:** `two_pass_llama_detector.py`
- **Logic:**
  ```python
  if question_type == "subjective":
      return UNCERTAIN  # Always abstain from subjective questions
  ```
- **Acceptance:** test_ambiguous shows accuracy ≥ 60% (up from 24%)

#### W2-T3: Nonsense Coherence Path
- **Assigned:** detector-engineer
- **Duration:** 4h
- **Dependencies:** W1-T2, W1-T3
- **Description:** When question-type = factual but coherence check fails, override to unknown
- **Changes:** `two_pass_llama_detector.py`
- **Logic:**
  ```python
  if question_type == "factual":
      coherence = coherence_check(question)
      if coherence < 0.3:  # Semantically incoherent
          return UNKNOWN
  ```
- **Acceptance:** test_nonsense shows unknown_rate ≥ 75% (up from 48%)

#### W2-T4: Niche Cluster Calibration
- **Assigned:** detector-engineer
- **Duration:** 6h
- **Dependencies:** W1-T1
- **Description:** Replace point-based embedding distance with cluster-based distance
- **Changes:** `two_pass_llama_detector.py` calibrate() and detect()
- **Approach:**
  - Use K-means clustering on known embeddings (10 clusters)
  - Use K-means clustering on unknown embeddings (10 clusters)
  - Distance = distance to nearest cluster center, not nearest point
- **Acceptance:** test_niche accuracy ≥ 60% (up from 33%)

#### W2-T5: Meta Calibration Set
- **Assigned:** detector-engineer
- **Duration:** 4h
- **Dependencies:** W1-T3
- **Description:** Add 20 meta-questions to calibration set
- **Changes:** `ep_dataset.py` (add meta_known category)
- **Acceptance:** test_meta shows consistency ≥ 80% (up from 44%)

---

### WAVE 3: Validation

#### W3-T1: Edge Case Re-Run
- **Assigned:** test-engineer
- **Duration:** 4h
- **Dependencies:** W2-T1, W2-T2, W2-T3, W2-T4, W2-T5
- **Description:** Run all 14 edge cases with hardened detector
- **Acceptance:**
  - Overall: ≥ 10/14 pass (up from 5/14)
  - Counterfactual: ≥ 70%
  - Nonsense: ≥ 75% unknown rate
  - Ambiguous: ≥ 60%
  - Niche: ≥ 60%
  - Meta: ≥ 80% consistency
  - Length: |correlation| < 0.3

#### W3-T2: Hallucination Attack Validation
- **Assigned:** test-engineer
- **Duration:** 6h
- **Dependencies:** W2-T1, W2-T2, W2-T3
- **Description:** Test the reverse exploitation: craft prompts that SHOULD be known but trick the detector
- **Attack prompts:**
  1. "What if gravity didn't exist?" (should be known → was unknown)
  2. "What is the color of Tuesday?" (should be unknown → was known)
  3. "What is the best programming language?" (should be uncertain → was known)
- **Acceptance:**
  - Attack 1 (counterfactual exile): Fixed → now correctly known
  - Attack 2 (syntactic camouflage): Fixed → now correctly unknown
  - Attack 3 (subjective injection): Fixed → now correctly uncertain

#### W3-T3: Report Generation
- **Assigned:** integration-lead
- **Duration:** 2h
- **Dependencies:** W3-T1, W3-T2
- **Description:** Generate before/after comparison report
- **Acceptance:** Report shows improvement on all 9 failing edge cases

---

## ACCEPTANCE CRITERIA SUMMARY

| Task | Criterion | Measure |
|------|-----------|---------|
| W1-T1 | Length correlation fixed | \|r\| < 0.3 on test_length |
| W1-T2 | Nonsense detection improved | unknown_rate ≥ 70% |
| W1-T3 | Question classifier accurate | ≥ 90% on 50 test questions |
| W1-T4 | Consistency check works | Counterfactual consistency ≥ 0.7 |
| W2-T1 | Counterfactuals fixed | accuracy ≥ 70% |
| W2-T2 | Subjective flagged | accuracy ≥ 60% on ambiguous |
| W2-T3 | Nonsense caught | unknown_rate ≥ 75% |
| W2-T4 | Niche detected | accuracy ≥ 60% |
| W2-T5 | Meta consistent | consistency ≥ 80% |
| W3-T1 | Overall improvement | ≥ 10/14 edge cases pass |
| W3-T2 | Attacks closed | All 3 exploitation prompts fixed |
| W3-T3 | Report complete | Before/after documented |

---

## RISK MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Self-consistency too slow (3x generation) | Medium | High | Cache results, skip for short questions |
| sentence-transformers too large for M1 | Low | High | Use `all-MiniLM-L6-v2` (80MB), fall back to rule-based |
| Cluster calibration overfits | Medium | Medium | Use K=10, validate on held-out niche questions |
| Question classifier has false positives | Medium | Medium | Conservative rules, flag uncertain instead of forcing type |
| Regression on existing CV | Low | High | Run W0-T2 before every wave |

---

## DEFINITION OF DONE

- [ ] All 12 tasks complete with acceptance criteria met
- [ ] No regression on 100% CV baseline
- [ ] ≥ 10/14 edge cases pass
- [ ] 3 hallucination attack vectors closed
- [ ] Code passes ruff + mypy
- [ ] Documentation updated
- [ ] PR merged to main

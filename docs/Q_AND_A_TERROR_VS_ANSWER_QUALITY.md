# Q&A: Does Less "Terror" Improve Answer Quality or Speed?

**Question:** If a question has less "terror," does that mean the answer is more accurate or generated faster?

**Short Answer:** No. Less "terror" means more reliable uncertainty detection, not better answers or faster generation.

---

## What "Less Terror" Actually Means

The "terror" metric (KL divergence) measures how much the question format disrupts the model's next-token distribution:

| Format | Mean KL | What Happens |
|--------|---------|-------------|
| WH-question | 0.14 | Clean, stable distribution — model knows what type of answer is expected |
| Statement | 0.51 | Disruptive — model must figure out grammar AND content simultaneously |

## Does Less Terror Improve Answer Accuracy?

**No.** For a known question, the answer is correct regardless of format. For an unknown question, the answer is hallucinated regardless of format. The "terror" is about the reliability of the uncertainty signal, not the answer quality.

Where it does matter: a low-terror format produces a cleaner signal for the detector. The entropy, norm, and embedding distance are more consistent across paraphrases, so the detector can more reliably distinguish "known" from "unknown." The model isn't fighting against an awkward grammatical frame.

## Does Less Terror Shorten Time-to-Answer?

**Not measured directly.** The framework only looks at the first token (`max_tokens=1`). But there is a plausible connection: a more peaked distribution (lower entropy, less terror) means the model is more decisive about its first token. In full generation, this might translate to fewer "hesitation" tokens. However, the forward pass time is dominated by model size, not distribution shape.

## The Real Benefit

The detector's job is to decide **whether to generate an answer at all.** A low-terror format makes that decision more reliable because the uncertainty signals (entropy, norm, format variance) are less noisy. The "terror" is terror for the detector, not for the model's answer quality.

# Trust & Safety / Content Moderation Datasets — Research Findings

Research date: 2026-09-27. Every dataset below was confirmed to exist via live web search / page fetch (not recalled from memory). Sizes and license claims are as stated on the dataset's own hosting page (Kaggle/Hugging Face/GitHub/UCI) at the time of this research — verify again before shipping, since Kaggle/HF metadata can drift.

---

## 1. Jigsaw Toxic Comment Classification Challenge

**Status:** Verified

**URL:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data (mirrors: https://huggingface.co/datasets/google/jigsaw_toxicity_pred, https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes)

**License:** Annotations released as **CC0 (public domain)** by Jigsaw/Google's Conversation AI team. The underlying comment text is Wikipedia talk-page content, governed by Wikipedia's **CC-BY-SA 3.0** — so redistribution of the raw text should carry attribution/share-alike consideration, but using it to train a model is standard practice and widely done commercially (e.g. Perspective API lineage).

**Size:** ~159,571 labeled comments in the train split (plus a ~63,978-row public test set with labels released post-competition) — roughly 223K rows total. Source: Wikipedia talk-page comments.

**Label schema:** 6 independent binary flags per comment: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` (multi-label, not mutually exclusive).

**Fit assessment:** Excellent `choice` fit if you bucket into a single dominant category, and a very natural `noul` fit as-is (six independent yes/no statements like "this comment is a threat"). Weaker `score` fit since labels are binary, not ordinal — would need to derive severity from label co-occurrence.

**Known quality issues:** Heavily skewed toward "not toxic" (roughly 90%+ negative class per label); English-only; annotator identity-bias concerns were the entire motivation for dataset #2 below (models trained on this dataset over-flag comments mentioning identity terms like "gay" or "muslim" regardless of tone); comments are Wikipedia talk-page style, not necessarily representative of social-media or product-review moderation contexts.

---

## 2. Jigsaw Unintended Bias in Toxicity Classification (Civil Comments)

**Status:** Verified

**URL:** https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data — underlying corpus mirrored at https://huggingface.co/datasets/google/civil_comments

**License:** **CC0 1.0** (public domain). Comments originally came from the defunct Civil Comments platform, which explicitly released ~2M comments into an open archive for research when it shut down.

**Size:** ~1.999M rows total (~1.8M train / ~97K validation / ~97K test on the HF mirror).

**Label schema:** Continuous **float 0–1 scores** (fraction of annotators who agreed) for `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, `sexual_explicit`, plus ~24 identity-mention subgroup columns (gender, race, religion, sexual orientation, disability, etc.) used for bias auditing.

**Fit assessment:** This is the strongest **`score`** fit in the list — genuinely continuous, calibrated severity per attribute, ideal for training/validating a calibrated P(true)-style or ordinal-severity head. Also supports `noul` well by thresholding (e.g. "P(toxicity > 0.5)"), and `choice` by taking argmax across the subtype columns. The identity-subgroup columns are valuable for building bias/fairness eval sets alongside training data.

**Known quality issues:** Same over-triggering-on-identity-terms bias problem that motivated the dataset's creation is still visible in the raw labels (annotators, not the platform, introduced this bias); English-only; comments skew toward US-centric news-comment discourse (Civil Comments was a commenting widget for news sites).

---

## 3. HateXplain

**Status:** Verified (license has a documentation inconsistency — see below)

**URL:** https://github.com/hate-alert/HateXplain (paper: AAAI 2021, "HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection"); HF mirror: https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain

**License:** The GitHub repo displays an **MIT License** badge/footer, but the Hugging Face dataset card metadata lists **CC-BY-4.0**, and dataset aggregators (e.g. DBLP) list **CC0**. This is a genuine inconsistency across the dataset's own distribution channels — flag as **uncertain / needs the maintainers to confirm** before commercial use, though MIT (the repo's own stated license) and CC-BY-4.0 both permit commercial use with attribution. Underlying posts are Twitter/Gab text, which may carry separate platform ToS constraints on redistributing raw post text at scale.

**Size:** ~20,148–20,229 posts (sources differ slightly): 9,055 from Twitter, 11,093–11,174 from Gab.

**Label schema:** 3-class label per post — `hatespeech` / `offensive` / `normal` — annotated by 3 annotators each, plus a `target` field naming the victimized community/communities, plus token-level binary `rationales` (which words justified the label).

**Fit assessment:** Strong `choice` fit for the 3-class label and for target-community classification; the token rationales are a bonus for explainability but not directly a question type. Could derive a coarse `noul` ("is this hate speech: yes/no") by collapsing hatespeech vs. {offensive, normal}. Not naturally ordinal, so weak `score` fit unless you treat hatespeech > offensive > normal as a severity ladder.

**Known quality issues:** Low inter-annotator agreement reported by the authors themselves (Krippendorff's α ≈ 0.42 for labels, 0.49 for rationales), meaning real annotator disagreement/noise; Gab-sourced content skews toward far-right extremist rhetoric, so class balance and topical distribution are not representative of general-purpose moderation; English-only; small by modern standards (~20K).

---

## 4. Davidson et al. — Hate Speech and Offensive Language Dataset

**Status:** Verified

**URL:** https://github.com/t-davidson/hate-speech-and-offensive-language (paper: Davidson, Warmsley, Macy & Weber, "Automated Hate Speech Detection and the Problem of Offensive Language," ICWSM 2017); mirrors on Kaggle (https://www.kaggle.com/datasets/eldrich/hate-speech-offensive-tweets-by-davidson-et-al) and HF (https://huggingface.co/datasets/tdavidson/hate_speech_offensive).

**License:** **MIT License** (confirmed via the repo's own LICENSE file) — clearly permits commercial use.

**Size:** ~24,783 tweets (commonly cited figure for `labeled_data.csv`).

**Label schema:** 3-class majority label per tweet — `hate_speech`, `offensive_language`, `neither` — plus raw per-tweet annotator counts (`hate_speech_count`, `offensive_language_count`, `neither_count`, `count` of annotators), which lets you reconstruct a soft/calibrated distribution rather than just the hard majority label.

**Fit assessment:** Very good `choice` fit (3-way category), and the retained annotator count columns make it one of the few classic datasets where you can derive a genuinely calibrated P(true) `noul` signal (agreement fraction) instead of a synthetic one. Weak native `score` fit (no ordinal severity), though hate > offensive > neither is a plausible severity ordering if needed.

**Known quality issues:** Well-documented racial bias — tweets in African-American English are disproportionately flagged as offensive/hate speech by the crowd annotators (widely discussed in follow-up fairness literature); small, English-only, Twitter-only (2017-era discourse, pre-dates a lot of modern slang/coded language); "neither" class is a catch-all that mixes truly benign tweets with merely crude-but-not-hateful ones.

---

## 5. Measuring Hate Speech (UC Berkeley D-Lab)

**Status:** Verified

**URL:** https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech

**License:** **CC-BY-4.0** — permits commercial use with attribution.

**Size:** ~135K–136K annotation rows (≈39,565–50,000 unique comments, each rated by a mean of ~4 annotators via Mechanical Turk; ~11K unique annotators) drawn from YouTube, Twitter, and Reddit.

**Label schema:** Ten ordinal (0–4 style) sub-dimensions per annotation — `sentiment`, `respect`, `insult`, `humiliate`, `status`, `dehumanize`, `violence`, `genocide`, `attack_defend`, plus a binary `hatespeech` judgment — which are combined via a many-facet Rasch/IRT model into a single continuous **`hate_speech_score`** that adjusts for individual annotator severity/leniency bias.

**Fit assessment:** Best-in-class `score` fit alongside dataset #2 — it is explicitly built as a continuous severity construct via IRT, which is methodologically close to what a calibrated severity head should learn. The ordinal sub-dimensions (insult, humiliation, dehumanization, violence, genocide) are individually useful `score` targets too. The binary `hatespeech` field is a ready-made `noul` target, and the sub-dimension set can be argmax'd into a `choice` label (e.g. dominant harm type).

**Known quality issues:** Multi-annotator design deliberately keeps raw disagreement (a feature, not a bug, for calibration purposes, but means naive majority-vote use throws away signal); English-only; comments are short (4–600 characters) which may not represent longer-form posts; platform mix (YouTube/Twitter/Reddit) is dated to the 2019–2021 collection window.

---

## 6. SMS Spam Collection Dataset

**Status:** Verified

**URL:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection (original UCI source); Kaggle mirror: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

**License:** **CC BY 4.0** — permits commercial use with attribution.

**Size:** 5,574 SMS messages (4,827 ham / 747 spam), assembled from four sub-sources (NUS SMS Corpus, Grumbletext UK forum, Caroline Tagg's PhD corpus, and SMS Spam Corpus v0.1 Big).

**Label schema:** Single binary label per message: `ham` or `spam`.

**Fit assessment:** Clean, simple `noul` fit ("is this message spam: yes/no"); trivially also a 2-way `choice`. No ordinal severity dimension, so no `score` fit without synthetic augmentation (e.g. confidence-of-spam-type).

**Known quality issues:** Small and dated (collected mid-2000s to early-2010s UK/Singapore SMS norms — spam patterns have shifted substantially since, e.g. no modern smishing/QR-code scam patterns); English-only with UK/Singaporean slang skew; class imbalance (~87% ham); message-level only, no metadata (sender, timestamp).

---

## 7. Enron-Spam Dataset (Metsis, Androutsopoulos & Paliouras, 2006)

**Status:** Verified for existence/size; **license Uncertain**

**URL:** https://github.com/MWiechmann/enron_spam_data (cleaned single-CSV mirror); also on Kaggle (e.g. https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data) and HF (https://huggingface.co/datasets/bvk/ENRON-spam).

**License:** **No explicit license is stated** in the maintained GitHub mirror's README. The underlying Enron corpus itself became public through the FERC/Enron litigation and is broadly treated as public-domain-ish for research reuse, and the spam side was assembled by the paper's authors from SpamAssassin, Project Honey Pot, and other sources of similarly ambiguous licensing. **Flag clearly: do not treat as cleared for commercial training without your own legal review** — this is a "widely used in academia, licensing never formalized" situation, not a confirmed-open dataset.

**Size:** 33,716 emails total — 17,171 spam / 16,545 ham — from six Enron employees' mailboxes plus spam pooled from multiple external sources.

**Label schema:** Binary `spam`/`ham` label, plus `Subject`, `Message`, and `Date` fields.

**Fit assessment:** Good `noul`/`choice` fit for email-specific spam detection (complements the SMS dataset with longer-form, subject+body text), but the license gap and age (emails are from 1999–2002 Enron correspondence plus circa-2005 spam samples) make it best used for prototyping only, not as a production commercial-training source until licensing is clarified.

**Known quality issues:** License ambiguity (above); very dated content (early-2000s email conventions and spam tactics, e.g. Nigerian-prince/phishing styles typical of that era, not modern spam); corporate-email domain (Enron) skews "ham" toward business correspondence, not representative of consumer inboxes.

---

## 8. OLID — Offensive Language Identification Dataset

**Status:** Uncertain (exists and is real, but access/license terms are not a clean open grant)

**URL:** Distributed via the OffensEval shared task site (https://sites.google.com/site/offensevalsharedtask/olid), used in SemEval-2019 Task 6 (Zampieri et al., NAACL 2019, "Predicting the Type and Target of Offensive Posts in Social Media").

**License:** Not clearly published as an open license. Access is via the shared-task distribution channel (originally a Google Form / task organizers), and third-party GitHub mirrors show "no license provided." Given it's raw tweet text, redistribution is also constrained by Twitter/X's developer terms on bulk-sharing tweet content. **Treat as research-only / unclear-for-commercial-use** until you can get explicit confirmation from the OffensEval organizers.

**Size:** 14,200 annotated English tweets.

**Label schema:** Hierarchical 3-level annotation: (A) offensive vs. not offensive, (B) targeted vs. untargeted insult, (C) target type — individual / group / other.

**Fit assessment:** Nice `choice` fit for the hierarchical target-type taxonomy (level C) and level A doubles as a `noul` ("is this offensive"). No ordinal dimension for `score`.

**Known quality issues:** License/access uncertainty (above) is the main blocker; small (14.2K); Twitter-only, 2019-era; later work (SOLID/OLID-BR etc.) found meaningful annotation inconsistencies at the target-identification level.

---

## Recommendation: best 2–3 starting points

1. **Jigsaw Unintended Bias in Toxicity Classification / Civil Comments (#2)** — the clear first pick. CC0, ~2M rows, and uniquely offers genuinely continuous 0–1 severity scores across 7 toxicity dimensions plus identity-subgroup metadata for bias auditing. This is the best direct training signal for the `score` question type and can be thresholded for `noul` or argmax'd for `choice`.

2. **Jigsaw Toxic Comment Classification Challenge (#1)** — CC0, ~223K rows, complements #2 by giving clean multi-label binary flags that map directly onto independent `noul` questions ("is this a threat," "is this an insult," etc.) and a natural `choice` category set. Same underlying corpus family as #2, so the two combine well without licensing friction.

3. **Measuring Hate Speech (#5)** — CC-BY-4.0, methodologically the most rigorous severity construct available (IRT-derived, bias-corrected `hate_speech_score`), and its ten ordinal sub-dimensions (insult, humiliation, dehumanization, violence, genocide) are a strong additional source of `score`-type training examples that #2 doesn't cover as granularly, particularly for the hate-speech-specific severity ladder rather than generic toxicity.

Together these three are all commercially-usable (CC0 / CC-BY-4.0), collectively cover choice + score + noul with real calibrated/soft labels (not just hard majority votes), and total well over 2 million labeled examples. Davidson (#4) and SMS Spam (#6) are good small, clean, MIT/CC-BY supplements once you need multiclass hate/offensive/neither coverage or a spam-specific slice; HateXplain (#3), Enron-Spam (#7), and OLID (#8) are useful for prototyping or explainability work but should be held back from production training until their license ambiguities are resolved.

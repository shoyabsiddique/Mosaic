# Public Datasets for Email/Communication Classification (System 1 Training Data)

Domain: intent classification, priority/urgency scoring, spam detection, phishing detection.
Target question types: `choice` (pick one label), `score` (ordinal priority), `noul` (yes/no statement with calibrated P(true)).

All datasets below were confirmed to exist via live web search on 2026-09-27. Sizes and license
claims are as reported on the dataset's own hosting page; verify the license text directly before
shipping anything commercial, since Kaggle re-uploads sometimes attach a looser license tag than
the original data source actually permits.

---

## 1. Enron-Spam Dataset (Metsis, Androutsopoulos & Paliouras, CEAS 2006)

**Status: Verified**

- **URL:** https://github.com/MWiechmann/enron_spam_data (clean single-CSV re-release of the
  original corpus); also mirrored as `SetFit/enron_spam` on Hugging Face
  (https://huggingface.co/datasets/SetFit/enron_spam) and as Kaggle dataset
  `bayes2003/emails-for-spam-or-ham-classification-enron-2006`.
- **License:** Underlying data derives from the FERC-released, public-domain Enron corpus. The
  GitHub re-release does not display an explicit LICENSE file in search results — **verify before
  commercial use**, but the source emails themselves carry no known commercial restriction.
- **Size:** 33,716 emails total (17,171 spam + 16,545 ham), drawn from 6 Enron employees'
  mailboxes (Farmer, Kaminski, Kitchen, Williams, Beck, Lokay).
- **Labels:** Binary `spam` / `ham` per email, plus original folder/employee metadata.
- **Fit:** Strong for `noul` ("this message is unsolicited/spam" — P(true)) and for `choice`
  (spam vs. ham). No native priority or intent labels.
- **Known issues:** ~2000s-era corporate email; spam patterns (Nigerian-prince era, pharma spam)
  are dated compared to modern phishing/spam tactics. English-only. Reasonably balanced (~51/49).

---

## 2. SpamAssassin Public Corpus

**Status: Verified**

- **URL:** https://spamassassin.apache.org/old/publiccorpus/ (original); also on Kaggle as
  `beatoa/spamassassin-public-corpus` and Hugging Face as `talby/spamassassin`.
- **License:** Confirmed — Open Data Commons Public Domain Dedication & License 1.0 for the data
  files, contents under CC0 1.0 Universal. **Commercial training is permitted.**
- **Size:** 6,047 messages total: 500 spam, 2,500 easy_ham, 250 hard_ham, 1,400 easy_ham_2,
  1,397 spam_2 (~31% spam ratio).
- **Labels:** Binary spam/ham, with a "hard_ham" tier (legitimate mail that looks spam-like) —
  useful as a built-in difficulty/calibration stratum.
- **Fit:** Good for `noul` (P(spam)) and `choice` (spam/easy-ham/hard-ham as three classes). The
  hard_ham split is particularly useful for calibration testing since it's designed to trip up
  naive classifiers.
- **Known issues:** Early-2000s mailing-list-era spam and Usenet-adjacent ham; small by modern
  standards; English-only; no phishing-specific labels (phishing is a subset of "spam" here at
  best).

---

## 3. Nazario Phishing Corpus

**Status: Verified (existence and size), Uncertain (license)**

- **URL:** https://monkey.org/~jose/phishing/ (original mbox files, collected by Jose Nazario).
  Frequently redistributed as CSV, e.g. in
  https://github.com/rokibulroni/Phishing-Email-Dataset/blob/main/Nazario.csv and folded into
  combined datasets (see #4 below).
- **License:** No formal license is published by the original curator; it's a personally
  collected inbox released for research use. **Treat as research/non-commercial unless you can
  get explicit clearance** — this is the single biggest licensing risk in this list given how
  widely it's redistributed without re-stated terms.
- **Size:** ~11,527 phishing email samples (positive class only — no legitimate/ham counterpart
  is included; it's typically paired with a ham source like SpamAssassin's easy_ham).
- **Labels:** Phishing-only (single class); redistributions sometimes add a synthetic negative
  class from another corpus.
- **Fit:** Only useful for `noul`/`choice` once paired with a negative (ham) set — on its own it
  cannot train a calibrated P(true) for "is this phishing" (no negative examples to calibrate
  against).
- **Known issues:** Age (corpus spans mid-2000s to ~2015 depending on snapshot) means it under-
  represents modern phishing techniques (OAuth-consent phishing, QR-code phishing, AI-generated
  lures). English-dominant.

---

## 4. Combined Phishing Email Dataset (CEAS + Nazario + Nigerian Fraud + SpamAssassin merge)

**Status: Verified (existence), Uncertain (exact size/license)**

- **URL:** https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- **License:** Displayed on the Kaggle page (not independently re-verifiable via search in this
  session — **check the license badge on the page directly before commercial use**). Associated
  paper: Al-Subaiey et al., "Novel Interpretable and Robust Web-based AI Platform for Phishing
  Email Detection" (2024), which requests citation.
- **Size:** Multiple source files (CEAS_08.csv, Nazario.csv, Nigerian_Fraud.csv, SpamAssassin.csv,
  plus a merged file); exact merged row count wasn't confirmable via search — expect tens of
  thousands based on constituent corpora (CEAS alone is ~39K messages per the original CEAS 2008
  challenge).
- **Labels:** Binary label column (phishing=1/legitimate=0) per row, with sender, receiver, date,
  subject, body, URLs — the URL field is a useful extra signal for a phishing `noul` question.
- **Fit:** Best out-of-the-box binary phishing dataset on this list because it already has a
  negative class merged in. Good for `noul` (P(phishing)) and `choice` (phishing vs. legitimate).
- **Known issues:** Mixed provenance/eras across the four constituent corpora means label
  definitions ("phishing" vs. "spam" vs. "fraud") aren't perfectly harmonized; re-verify license
  before commercial use since it aggregates several differently-licensed sources.

---

## 5. Multilingual Customer Support Tickets (Tobi Bueck)

**Status: Verified**

- **URL:** https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
  (also on Hugging Face as `Tobi-Bueck/customer-support-tickets`).
- **License:** Confirmed — CC BY 4.0. **Commercial use permitted with attribution.**
- **Size:** 28,587 records (per the Kaggle version referenced in search results).
- **Labels:** `priority` (1=Low, 2=Medium, 3=Critical), `queue` (department/category — e.g.
  billing, technical, general), `type`, `language` (English, German, Spanish, Portuguese,
  French), plus subject/body/agent-answer text.
- **Fit:** The best match on this list for the `score` question type — priority is a native
  3-level ordinal label. `queue`/`type` maps directly to `choice` (intent/category
  classification). Multilingual coverage is also a plus most email-specific datasets lack.
- **Known issues:** These are support tickets, not inbox email per se (though textually very
  similar — subject + body + free text). Some ticket text may be synthetically generated/
  augmented (per the dataset's own "(Synthetic)" companion release) — check which version/split
  you're pulling from, since a purely synthetic split will not reflect real user phrasing
  distributions as reliably as organic data.

---

## 6. SMS Spam Collection (UCI / Almeida & Hidalgo)

**Status: Verified**

- **URL:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection (also
  `ucirvine/sms_spam` on Hugging Face and `uciml/sms-spam-collection-dataset` on Kaggle).
- **License:** Confirmed — CC BY 4.0. **Commercial use permitted with attribution.**
- **Size:** 5,574 messages (spam/ham labeled), ~365 KB.
- **Labels:** Binary spam/ham.
- **Fit:** Useful as a secondary, cleanly-licensed binary spam source for `noul`/`choice`, but
  it's SMS text, not email — shorter, more informal register, no subject line/headers. Best used
  as supplementary data to broaden "unsolicited message" pattern coverage rather than as a
  primary email dataset.
- **Known issues:** Small; SMS register differs materially from email (no threading, headers, or
  HTML); somewhat dated (mid-2000s-2011 UK/Singapore-sourced messages).

---

## 7. Full Enron Email Corpus (raw, unlabeled)

**Status: Verified (existence), Not directly usable without additional labeling**

- **URL:** https://www.kaggle.com/datasets/wcukierski/enron-email-dataset (canonical Kaggle
  mirror); original release via FERC/CMU (https://www.cs.cmu.edu/~enron/).
- **License:** Source data is public domain (FERC investigative release). The Kaggle page's own
  license badge wasn't independently confirmed in this session — check it, but the underlying
  data has no known commercial restriction.
- **Size:** ~500,000 emails from ~150 Enron employees.
- **Labels:** None — raw mailbox dump organized by employee/folder only.
- **Fit:** Not directly usable for any of the three question types as-is (no intent/priority/
  spam labels). Only valuable as (a) a large pool of realistic corporate email *text* to sample
  states from, or (b) a base corpus to weakly-label yourself (e.g., folder name as a weak intent
  proxy) — not a verified labeled dataset in its own right, so it's listed for completeness
  rather than as a ready training set.
- **Known issues:** ~2000-2002 corporate email register; not representative of modern spam/
  phishing/consumer-intent patterns; privacy-sensitive (real people's real correspondence) even
  though publicly released.

---

## Recommendation

Start with these three:

1. **Multilingual Customer Support Tickets (#5)** — the only dataset here with a native ordinal
   `priority` label plus a `queue`/`type` category label, cleanly licensed (CC BY 4.0), and
   reasonably large (28.5K rows). This should be the backbone for the `score` (priority/urgency)
   and `choice` (intent/category) question types.
2. **SpamAssassin Public Corpus (#2)** — smallest license risk (CC0/public domain, fully
   confirmed), with a built-in "hard_ham" difficulty tier that's genuinely useful for testing
   calibration on ambiguous cases, not just easy separability. Use for the `noul` spam-detection
   question.
3. **Combined Phishing Email Dataset (#4)** — the most practical phishing source because it
   already merges positive (phishing) and negative (legitimate) examples across four source
   corpora, avoiding the "positive-only" problem that plagues the raw Nazario corpus. Re-verify
   its Kaggle license badge before commercial training, and treat the Nigerian_Fraud/CEAS/
   Nazario/SpamAssassin sub-splits as separate strata for calibration analysis given their
   different eras and styles.

Enron-Spam (#1) and SMS Spam Collection (#6) are good supplementary/robustness-check sets once
the above three are in place. Treat the raw Nazario corpus (#3) and the full unlabeled Enron
corpus (#7) as lower priority — the former needs pairing with a negative class you already get
for free in #4, and the latter has no labels at all.

All of the above are English-dominant except the multilingual support-ticket set; none of them
represent current (2025-2026) phishing/social-engineering tactics well, since they range from
~2000s (Enron, SpamAssassin, Nazario) to ~2024 (the combined phishing merge and the ticket
dataset). Plan to supplement with synthetic or freshly-collected modern examples for phishing
specifically, since attacker tactics have moved on substantially since these corpora were
assembled.

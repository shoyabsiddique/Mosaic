# Security Operations / Alert Triage — Candidate Training Datasets

Research date: 2026-09-27. All candidates below were checked via live web search against
their actual hosting pages (NIST, GitHub, Hugging Face, Kaggle, arXiv) — not recalled from
memory. "Verified" means I found a real, currently-reachable source page confirming
existence, approximate size, and license. "Uncertain" means some detail (usually exact
license terms or exact row count) could not be confirmed from what the search/fetch
returned, even though the dataset itself is real.

---

## 1. NVD (National Vulnerability Database) — CVE + CVSS data via API/JSON

**Status: Verified**
**URL:** https://nvd.nist.gov/developers/vulnerabilities (REST API); raw CVE records also
mirrored as JSON on GitHub, e.g. https://github.com/olbat/nvdcve and
https://github.com/password123456/nvd-cve-database
**License:** U.S. government work — public domain under Title 17 U.S. Code. No fee, no
registration required for the API (an optional API key just raises the rate limit).
NIST asks for attribution ("This product uses data from the NVD API but is not endorsed
or certified by the NVD") but this is a courtesy notice, not a use restriction. Fully
usable for commercial model training. Note: NVD retired the old bulk XML "data feeds" in
late 2023 in favor of the REST API — plan around the API, not the old feed format.
**Size:** ~280,000+ CVE records as of 2026 and growing daily; full historical corpus back
to CVE-1999-0001.
**Labels / schema:** Free-text vulnerability description (short, technical), CVSS v2/v3/v4
base score + severity band (NONE/LOW/MEDIUM/HIGH/CRITICAL), CWE weakness ID(s), affected
CPE/product list, reference URLs.
**Fit:**
- `score`: excellent — CVSS severity band is a natural ordinal label directly tied to the
  description text.
- `choice`: good — CWE category (vulnerability class) is a natural multi-class choice
  field, though CWE has hundreds of fine-grained classes so you'd want to bucket into a
  smaller taxonomy.
- `noul`: weak on its own (no natural yes/no statements), but usable if you synthesize
  statements like "this CVE affects a web application" from CPE/description parsing.
**Known quality issues:** Descriptions are short, technical, boilerplate ("A vulnerability
in [product] allows [actor] to [impact] via [vector]") rather than natural incident
narratives — good for vulnerability-severity scoring, not representative of how an analyst
writes up an alert or incident.

---

## 2. CIRCL Vulnerability-Lookup / VulnTrain CVSS dataset (Hugging Face)

**Status: Verified**
**URL:** https://huggingface.co/datasets/CIRCL/vulnerability-scores (full dump, ~779,178
rows) and a filtered derivative https://huggingface.co/datasets/AgileRLArena/vulnerability-scores-cvss-v3
(565,569 rows, 161MB, keeps rows with a valid combined CVSS v3.1 score)
**License:** CC-BY-4.0 (both the CIRCL source and the derivative) — permits commercial use
with attribution.
**Size:** ~565K–779K rows depending on filtering.
**Labels:** Input = vulnerability `description` (same NVD-style text); targets =
`severity_band` (categorical: NONE/LOW/MEDIUM/HIGH/CRITICAL) and
`cvss_v3_1_v3_combined` (continuous 0–10 score).
**Fit:**
- `score`: excellent, this is essentially NVD's data pre-packaged specifically for
  severity classification/regression training — saves you the CVE-API scraping/join work.
- `choice`: not directly (no CWE/category field in this particular cut — pair with #1 or
  #3 for CWE labels).
- `noul`: weak, same caveat as #1.
**Known quality issues:** Same terse/technical text as NVD, since it's sourced from NVD/
CVE descriptions.

---

## 3. CVE → CWE mapping datasets (Hugging Face, multiple)

**Status: Verified**
**URL:** e.g. https://huggingface.co/datasets/regularpooria/CVE_CWE_Software_Mapping_Dataset,
https://huggingface.co/datasets/stasvinokur/cve-and-cwe-dataset-1999-2025,
https://huggingface.co/datasets/xamxte/cve-to-cwe
**License:** Varies by uploader — check each dataset card individually before training;
most are derived from NVD (public domain) but the uploader's own curation/labels may carry
their own license (some are CC-BY, some unspecified — treat "unspecified" as "ask before
commercial use").
**Size:** stasvinokur's covers the full NVD history (CVE-1999-0001 through mid-2025, i.e.
~270K+ rows); regularpooria's and xamxte's are smaller curated subsets.
**Labels:** CVE-ID, CVE description, CWE-ID, CWE description/name, CWE parent category;
xamxte's variant also adds AI-assisted MITRE ATT&CK technique labels.
**Fit:**
- `choice`: excellent — CWE category (or its parent/rollup category) is a clean
  multi-class label paired directly with description text, better packaged for this use
  than parsing raw NVD JSON yourself.
- `score`: good if the specific dataset also carries CVSS (some do, check per-dataset).
- `noul`: weak, same as #1.
**Known quality issues:** Several of these are unofficial, community-curated re-packagings
of NVD data with "AI-assisted" label cleanup — spot-check label quality before relying on
them, and verify the license on the specific dataset card since these aren't
first-party NIST releases.

---

## 4. VERIS Community Database (VCDB)

**Status: Verified**
**URL:** https://github.com/vz-risk/VCDB (schema: https://github.com/vz-risk/veris)
**License:** CC-BY-SA 4.0 — commercial use permitted, but ShareAlike applies to
redistributed derivatives of the dataset itself (attribution + share-alike, not a
restriction on using it to train a private model, but worth having counsel confirm if you
plan to redistribute any dataset derivative).
**Size:** ~9,800 coded incidents (as of the last README snapshot checked), spanning
2013–2018+ publicly reported breaches; this is the same corpus underlying Verizon's DBIR.
**Labels:** Each incident is coded in the VERIS schema: threat actor (external/internal/
partner), action category (malware/hacking/social/misuse/physical/error/environmental),
asset variety, confidentiality/integrity/availability attributes, victim industry (NAICS),
incident timeline, plus a source-URL/analyst-notes field pointing to the original public
breach report.
**Fit:**
- `choice`: excellent — action category, asset variety, and industry are all natural,
  pre-coded multi-class fields tied to a real incident, closest analog to "incident
  category classification" in the prompt.
- `score`: moderate — no single ordinal severity field, but records/data volume and
  impact fields could be bucketed into an ordinal severity proxy.
- `noul`: moderate — could derive yes/no statements from the coded booleans (e.g. "this
  incident involved an external actor"), but the dataset wasn't built with noul-style
  statements in mind.
**Known quality issues:** Coded from public breach reports (not raw SOC narratives), so
narrative depth varies a lot by incident; VERIS coding is done by community volunteers so
label consistency/quality is uneven across contributors. Still one of the only real,
open, incident-level (not just vulnerability-level) security datasets that exists.

---

## 5. Microsoft GUIDE dataset (real SOC alert/incident triage labels)

**Status: Verified**
**URL:** Kaggle: https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction
(companion paper: arXiv 2407.09017, "AI-Driven Guided Response for Security Operation
Centers with Microsoft Copilot for Security")
**License:** CDLA-Permissive-2.0 (Community Data License Agreement) — designed to permit
commercial use.
**Size:** This is the big one for the domain you're targeting: 13M+ data points across 33
entity types, 1.6M alerts, 1M+ incidents, from 6,100+ real organizations, with
customer-assigned triage labels, covering 441 MITRE ATT&CK techniques and ~9,100 distinct
detector IDs.
**Labels:** `IncidentGrade` — the real SOC analyst's triage verdict: True Positive /
Benign Positive / False Positive. Also `Category` (alert category), `MitreTechniques`,
`DetectorId`, `ProductId`, `Severity`, `EntityType`, `EvidenceRole`, `AlertTitle`, plus ~67
engineered numeric feature columns per alert/incident.
**Fit:**
- `noul`: excellent and directly on-point — "is this alert a true positive" is exactly
  the false-positive-detection use case named in the brief, and this is real analyst-
  labeled ground truth at huge scale, which is normally very hard to get publicly.
- `choice`: good — alert `Category` and MITRE technique are natural multi-class choice
  targets.
- `score`: moderate — `Severity` field exists but is coarse (vendor-assigned severity
  tier, not a rich ordinal scale).
**Known quality issues (important):** Despite the name "GUIDE," most of the released
columns are IDs/categoricals/engineered numeric features (OrganizationId, DetectorId,
ProductId, 67 numeric columns) rather than rich free-text narratives — `AlertTitle` and
`Category` are the main text-ish fields, so you'll likely need to construct a
pseudo-narrative "state" string from the structured fields rather than finding ready-made
incident prose. Also flagging: a public reproduction effort (GitHub PR from an unrelated
2026 security-internship project) reported finding **incident-level label leakage** in
GUIDE during their verification pass — worth a data-leakage audit (e.g., checking that
train/val/test splits don't leak `IncidentId`-correlated rows) before trusting held-out
metrics on it.

---

## 6. MITRE TRAM / Center for Threat-Informed Defense training data

**Status: Verified**
**URL:** https://github.com/center-for-threat-informed-defense/tram
**License:** Apache License 2.0 — permits commercial use.
**Size:** MITRE's own published training corpus is modest: ~11,300 annotated
sentences/phrases pulled from public CTI reports, covering the 50 most common ATT&CK
techniques (out of 625+ total techniques — long tail is not covered).
**Labels:** Each sentence is multi-label annotated with the ATT&CK technique(s) it
describes (choice-style, though multi-label rather than strictly single-choice).
**Fit:**
- `choice`: good — genuine natural-language sentences (not terse CVE-style text) mapped
  to attack-technique categories, which is closer to "incident narrative" prose than the
  NVD-family datasets.
- `score` / `noul`: not a natural fit — no severity or yes/no statements in this corpus.
**Known quality issues:** Small (11K sentences), sentence-level rather than full-incident-
level, and only covers 50/625 techniques, so class coverage is narrow. Best used as a
supplementary source for `choice` questions about attack-technique/category
classification, not as a primary corpus.

---

## 7. Phishing/spam email corpora (Enron + Nazario + SpamAssassin + CEAS, combined)

**Status: Verified**
**URL:** https://huggingface.co/datasets/ealvaradob/phishing-dataset (a standardized
combination of Enron legitimate email, Nazario phishing corpus, SpamAssassin, CEAS 2008,
Ling-Spam, and a Nigerian-fraud corpus — a similar combined 82,255-email corpus is
described in arXiv 2507.17978 "MeAJOR Corpus")
**License:** Apache 2.0 on the Hugging Face packaging; underlying source corpora
(Enron corpus, SpamAssassin public corpus, Nazario corpus) have each been used
commercially and academically for two decades without restriction, but note these are
older research corpora assembled from real (if dated) email traffic — worth a light
compliance check if used to train a product feature rather than just a research model.
**Size:** ~10K–100K samples depending on which sub-slice you pull (18,000+ raw emails,
plus SMS/URL/HTML sub-datasets bundled in the same release; combined multi-source corpora
in the literature reach ~82K emails).
**Labels:** Binary phishing/benign (or spam/ham) label per message; genuine natural-
language email body text.
**Fit:**
- `noul`: good — "this message is a phishing attempt" is a natural calibrated yes/no
  statement, and this is one of the few security-adjacent domains with abundant, truly
  natural-language text (full email bodies, not terse CVE strings).
- `choice`: moderate — could extend to phishing-technique or attack-vector choice labels
  with additional annotation, but the base datasets are binary.
- `score`: weak — no ordinal severity concept here.
**Known quality issues:** This is phishing/email-security, adjacent to but not the same
as "SOC/SIEM alert triage" — it's a reasonable proxy for the noul "is this a true
positive" pattern and for natural-language security text generally, but doesn't map
directly to vulnerability or incident-category taxonomies. Some source corpora (Nazario,
early SpamAssassin) are 15–20 years old, so vocabulary/attack patterns are dated relative
to current phishing.

---

## Datasets investigated and rejected as poor fits (flagging, not padding the list)

- **CICIDS2017** (Canadian Institute for Cybersecurity) and **UNSW-NB15**: both real and
  well-documented (CICIDS2017: ~2.8M labeled network flow records across 7 attack
  categories via CICFlowMeter, 80 flow features; UNSW-NB15: ~49/42 features across 9
  attack families), and both are free for research use — **but confirmed to be almost
  entirely numeric network-flow features** (packet counts, byte counts, inter-arrival
  timing, protocol flags), not text. There is no natural-language description field to
  put into a text/JSON "state" for this model; you would have to synthesize narrative text
  from the numeric features yourself, which defeats the purpose of using a "real" text
  dataset. Not recommended as-is for a text/JSON-driven typed-decision model — only
  useful if you build a templated narrative-generation step on top, which introduces
  synthetic text rather than real analyst language.
- **Public SOC-alert-triage / false-positive datasets from academic papers**: confirmed
  the concern in the brief — these are indeed hard to find publicly. Search turned up
  mostly (a) small GitHub demo projects built on synthetic or OTRF Security-Datasets
  attack-simulation logs (e.g. `soc-triage-lab`, `ai-alert-triage`), (b) a private
  1.27M-event South Korean SOC dataset used in one academic paper but not released
  publicly, and (c) the CORTEX dataset (multi-class triage-outcome labels: Actionable /
  Benign Positive / False Positive-Logic / False Positive-Data / Undetermined) described
  in an arXiv paper (2510.00311) whose public data-release status I could not confirm from
  search alone — treat CORTEX as **Uncertain** and verify direct availability before
  relying on it. The Microsoft GUIDE dataset (#5 above) is the one genuinely large, real,
  publicly-released exception to "SOC triage data is sensitive/unavailable," which is why
  it's the standout recommendation below.

---

## Recommendation: best 2–3 to start with

1. **Microsoft GUIDE dataset (#5)** — the closest match to the stated goal (SOC/SIEM
   alert triage, false-positive detection) and the only large-scale, real,
   analyst-labeled, publicly released dataset of its kind. Use `IncidentGrade`
   (TP/BP/FP) to train the `noul` "is this a true positive" head, and `Category`/
   `MitreTechniques` for `choice` heads. Budget time for (a) constructing a text/JSON
   "state" representation from its mostly-structured columns since raw narrative text is
   thin, and (b) auditing for the reported label-leakage issue before trusting metrics.

2. **NVD CVE/CVSS data (#1), ideally via the pre-packaged CIRCL/VulnTrain cut (#2) and a
   CVE→CWE mapping (#3)** — together these give you the best `score` (CVSS severity) and
   `choice` (CWE category) training signal for vulnerability-severity scoring, backed by
   the largest, cleanest, unambiguously public-domain text corpus in the space. Combine
   the three rather than picking one: #2 saves you the API-scraping work for severity, and
   #3 adds the CWE category axis that #2 alone lacks.

3. **VERIS Community Database (#4)** — the best source for genuine incident-category
   `choice` labels (actor/action/asset/industry) tied to real, named breaches rather than
   vulnerability descriptions, filling the "incident category classification" leg of the
   brief that NVD-family data doesn't cover. Treat MITRE TRAM (#6) and the phishing corpus
   (#7) as smaller supplementary sources rather than primary ones — TRAM adds
   natural-language attack-technique text at small scale, and the phishing corpus adds a
   genuine `noul` "is this real" pattern in fluent natural-language text, useful for
   diversifying training data beyond the terse CVE style.

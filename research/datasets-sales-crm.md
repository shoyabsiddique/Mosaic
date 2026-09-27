# Sales / CRM Domain — Public Dataset Survey

Scope: lead scoring, buying-intent detection, deal-stage classification, for training a "System 1" typed-decision model (choice / score / noul questions over a state text/JSON).

Bottom line up front: **the "thin data" assumption holds, but not for the reason usually assumed.** There are plenty of *Kaggle toy datasets* for lead scoring and CRM pipelines — they're small, tabular, and reasonably well-known in the ML-tutorial world (not genuinely obscure). What's actually scarce is (a) anything with real B2B **text** (notes, transcripts, emails) paired with calibrated labels, and (b) anything licensed for commercial use — the best text/transcript resources (CRMArena-Pro, CallCenterEN, the willingness-aware sales-talk corpus) are all research-only (CC-BY-NC). Tabular lead/deal data is available and commercially usable; text-grounded sales dialogue with usable labels, under a commercial-friendly license, essentially does not exist publicly at any scale.

---

## 1. CRM Sales Opportunities (Maven Analytics pipeline dataset)

**Status: Verified**
**URL:** https://mavenanalytics.io/data-playground/crm-sales-opportunities (original); mirrored on Kaggle by multiple uploaders, e.g. https://www.kaggle.com/datasets/innocentmfa/crm-sales-opportunities , https://www.kaggle.com/datasets/nilkamalsaha/crm-sales-opportunities-on-google-sheets , https://www.kaggle.com/datasets/mdasifikbalmamun/crm-sales-opportunities
**License:** Maven Analytics' Data Playground datasets are distributed as free public-domain practice data (per Maven's site). The individual Kaggle re-uploads don't consistently display an explicit license field in what's crawlable — treat as "verify per-mirror before commercial use," but the original source is intended for open reuse including commercial analytics practice.
**Size:** 4 linked CSVs — `sales_pipeline.csv` (~8,800 opportunities: 8,800 rows per multiple independent tutorials, e.g. GitHub analyses citing "8,801 rows"), `accounts.csv`, `products.csv`, `sales_teams.csv`.
**Labels:** `deal_stage` ∈ {Prospecting, Engaging, Won, Lost}; `close_value` (deal amount); `account`, `product`, `sales_agent`, `close_date`.
**Fit assessment:** Good structural fit for **choice** (deal_stage: 4-way) and **noul** (P(deal will close won) — binary Won/Lost is a natural calibration target). Weak fit for **score** unless you bucket `close_value` into ordinal tiers yourself. Biggest limitation: it's a fictitious hardware-reseller company with clean synthetic-feeling records, no free text (no notes/transcripts), so it teaches structure but not language grounding.

## 2. Lead Scoring Dataset ("X Education", Kaggle)

**Status: Verified**
**URL:** https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset (also mirrored as https://www.kaggle.com/datasets/ashydv/leads-dataset and used in the classic Upgrad/IIIT-B "Lead Scoring Case Study")
**License:** Not confirmed from crawlable metadata (Kaggle page did not expose it to fetch tools). Widely reused in tutorials/case studies without licensing disputes, but you should click through and confirm the Kaggle license badge yourself before commercial ingestion.
**Size:** 9,240 rows × 37 columns.
**Labels:** `Converted` (binary — lead became a customer or not); rich categorical features (Lead Source, Last Activity, Specialization, City, Occupation, Lead Quality, Asymmetrique Activity/Profile Index+Score).
**Fit assessment:** Strong fit for **noul** (P(this lead converts)) and reasonable for **score** — the dataset already ships an ordinal "Lead Quality" and Asymmetrique index/score fields that map naturally onto an ordinal fit/priority scale. Not text-grounded (all structured/categorical fields, education-vertical only, not general B2B).

## 3. CRM Sales Opportunities variants / "200K Customers" and similar Kaggle CRM tables

**Status: Verified (exist), Uncertain (usefulness)**
**URLs:** https://www.kaggle.com/datasets/mohamedramadan2040/customer-information-for-sales-targeting-and-crm ; https://www.kaggle.com/datasets/sushicatsan/sample-sales-crm-data ; https://www.kaggle.com/datasets/agungpambudi/crm-sales-predictive-analytics
**License:** Uncertain per-dataset — not independently confirmed.
**Size:** Varies (the "200K Customers" one is ~200K rows of customer/contact records).
**Labels:** Mostly customer demographic/segmentation fields, not lead/deal outcome labels.
**Fit assessment:** Low direct fit — these are customer-master tables, not decision-labeled lead/deal records. Useful only as a source of realistic *state* fields (company size, industry, contact role) to seed synthetic generation, not as label sources.

## 4. Online Shoppers Purchasing Intention Dataset (UCI)

**Status: Verified**
**URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset (mirrored on Kaggle: https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
**License:** CC BY 4.0 — permits commercial use with attribution.
**Size:** 12,330 sessions, 17 features + target.
**Labels:** `Revenue` (binary — session ended in purchase or not); page-visit/behavioral features (ProductRelated pages/duration, BounceRates, ExitRates, PageValues, month, visitor type).
**Fit assessment:** Good license, real and well-documented, but this is **B2C web-session behavior**, not CRM/B2B lead data — no company/contact fields, no free text. Useful as a calibration-quality reference (it's genuinely used in probability-calibration literature) and as a `noul` template ("will this visitor purchase"), but off-domain for B2B sales/CRM specifically.

## 5. Sales Conversion Optimization / Facebook Ad Clicks-Conversion Tracking (Kaggle)

**Status: Verified**
**URL:** https://www.kaggle.com/datasets/loveall/clicks-conversion-tracking
**License:** CC BY-NC-SA 4.0 — **non-commercial**, disqualifies it for commercial model training without a separate agreement.
**Size:** ~1,143 rows (small; ad-campaign-level, not lead-level).
**Labels:** Total_Conversion (inquiries), Approved_Conversion (purchases) — counts, not per-lead labels.
<br>**Fit assessment:** Poor fit even ignoring license — it's aggregated ad-spend data, not individual lead/deal records with typed labels.

## 6. CRMArena / CRMArena-Pro (Salesforce AI Research)

**Status: Verified**
**URL:** https://github.com/SalesforceAIResearch/CRMArena ; https://huggingface.co/datasets/Salesforce/CRMArenaPro ; paper: https://arxiv.org/abs/2505.18878
**License:** CC BY-NC-4.0 — **research only, not for commercial use.** Explicitly stated in the repo.
**Size:** CRMArena-Pro ≈ 8,614 examples/tasks (HF size category 1K–10K); 19 expert-validated task types across sales, service, and CPQ workflows, B2B and B2C.
**Labels:** Task-based (agent must query/act on a simulated CRM), not directly a choice/score/noul label set, but the underlying synthetic Salesforce-schema CRM records (accounts, opportunities, cases, leads) with realistic B2B/B2C business logic are exactly the kind of *state* structure this project wants.
**Fit assessment:** This is the single best-matched dataset in *design* (realistic CRM schema, sales-specific business scenarios, both B2B/B2C) but is **blocked by license** for commercial training. Worth studying its schema and task design as a template for synthetic generation, not worth ingesting directly for a commercial model.

## 7. SalesTranscriptQA (EndgameLabs)

**Status: Uncertain**
**URL:** https://github.com/Endgame-Labs/SalesTranscriptQA ; https://huggingface.co/datasets/EndgameLabs/SalesTranscriptQA
**License:** Not independently confirmed; repo is explicitly flagged "specification stage," i.e. not a finished/stable release.
**Size:** ~200 QA pairs (50 single-call + 50 two-call, for B2B and B2C each) — small, and it's a QA benchmark derived from CRMArena-Pro's synthetic transcripts, so it likely inherits CRMArena-Pro's CC-BY-NC restriction.
**Fit assessment:** Not usable as-is (too small, likely non-commercial, still in spec stage). Not recommended.

## 8. CallCenterEN (real-world call center transcripts)

**Status: Verified**
**URL:** https://arxiv.org/abs/2507.02958 ; https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english
**License:** CC BY-NC-4.0 — **non-commercial only.**
**Size:** 91,706 conversations / 10,448 audio hours (largest known open call-center transcript release), PII-redacted, US/India/Philippines accents.
**Labels:** None built-in for lead scoring/intent/deal-stage — it's raw agent-customer transcripts (customer support and sales mixed), would need labeling.
**Fit assessment:** Genuinely the most realistic large-scale **conversational** text resource found, and it's real (not synthetic) — but (a) non-commercial license blocks direct use, (b) it's not sales-labeled out of the box (mixed support/sales calls), (c) needs an LLM-labeling pass to get choice/score/noul targets even if license were resolved.

## 9. User Willingness-aware Sales Talk Dataset (CyberAgentAILab, COLING 2025)

**Status: Verified**
**URL:** https://github.com/CyberAgentAILab/salestalk-dataset ; paper: https://arxiv.org/abs/2412.19490
**License:** CC BY-NC-SA 4.0 — **non-commercial.**
**Size:** 109 dialogues, 3,289 utterances — small, and **Japanese only**.
**Labels:** This is the best label-schema match found: pre/post-dialogue purchase-intention rating (maps to **noul** P(will buy)), plus per-utterance willingness-to-continue / willingness-to-share-information / willingness-to-accept-objective, each rated positive/neutral/negative (maps well to a **score**-style ordinal signal per turn).
**Fit assessment:** Excellent schema template for exactly the "calibrated P(true) on a yes/no buying statement" pattern this project wants — but too small, wrong language, and non-commercial. Best used as a *design reference* for your own synthetic-data label taxonomy, not as training data.

## 10. SaaS Sales Conversations (DeepMostInnovations, Hugging Face)

**Status: Verified (exists), synthetic**
**URL:** https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations
**License:** Apache 2.0 — **commercial use permitted.**
**Size:** 100,000 rows, ~7.17 GB (includes precomputed 3,072-dim embeddings + conversation metadata).
**Labels:** Binary conversion outcome (0/1), plus customer engagement score, sales-effectiveness rating, and turn-by-turn conversion-probability trajectory.
**Fit assessment:** Best commercial-license match found for conversational sales data with graded/calibrated-style outputs (the turn-by-turn probability trajectory is directly analogous to a `noul` calibration target). Caveats: entirely LLM-generated/synthetic (created for a companion RL paper, "SalesRLAgent"), SaaS-vertical only, and quality/realism of the underlying dialogues is unverified beyond the authors' own claims — treat as a possible **augmentation** source, not a ground-truth source, and spot-check a sample before relying on its labels.

## 11. goendalf666/sales-conversations family (Hugging Face)

**Status: Verified (exists), weak fit**
**URL:** https://huggingface.co/datasets/goendalf666/sales-conversations (+ `-2`, `-instruction-base`, `-instruction-customer` variants; companion `sales-textbook_for_convincing_and_selling`)
**License:** Not clearly stated in what's crawlable — verify before use.
**Size:** ~3,410 rows (small-to-mid).
**Labels:** None — these are GPT-generated synthetic sales dialogues (inspired by the "Textbooks Are All You Need" methodology) intended for fine-tuning a sales chatbot's *generation*, not for classification/calibration.
**Fit assessment:** Not useful for this project — no choice/score/noul labels of any kind, and license is unclear.

## 12. Synthetic B2B CRM & Marketing Dataset (Kaggle)

**Status: Uncertain**
**URL:** https://www.kaggle.com/datasets/ezogngrd/synthetic-b2b-crm-and-marketing-data
**License:** Not confirmed (page content wasn't retrievable via available fetch tools; rate-limited scraping tool before this could be re-checked).
**Size/labels:** Unconfirmed — described in listings as "realistic CRM-style synthetic data with controlled noise." Worth a manual look before deciding, but cannot be verified as fit-for-purpose from this pass.

## 13. Deal/JDDC-style e-commerce customer service corpora (for reference only)

**Status: Verified (exists), off-domain**
**URL:** JDDC Corpus — https://arxiv.org/abs/1911.09969 (Chinese, e-commerce customer service, 289 intents)
**License:** Research-release terms apply (not general commercial license); also Chinese-language and customer-service rather than B2B-sales framed.
**Fit assessment:** Not recommended as a direct source; mentioned because it's the kind of large intent-labeled conversational corpus people mean when they say "surely something like this exists for sales" — it exists, but for a different vertical/language/license.

---

## Does "thin B2B CRM data" hold up?

Yes, with a specific shape to the gap:

- **Tabular lead/deal data**: not thin. Several genuine, reasonably-sized (thousands of rows), commercially-reusable-license (or license-ambiguous-but-widely-reused) datasets exist (#1, #2, #4). These give you real distributions for `choice` (deal_stage) and `noul` (converted/not) style targets, but zero free text.
- **Real B2B sales conversation/transcript text with labels**: genuinely thin. The few real, large, well-documented corpora that exist (CRMArena-Pro, CallCenterEN, the willingness-aware sales-talk corpus) are all non-commercial licensed by design (they're academic releases), and the one commercially-licensed conversational dataset found (#10) is fully synthetic and unverified for realism.
- **Deal-stage-labeled text notes** (the CRM-note-plus-stage-label combination that would most directly train this model): not found publicly at all. No dataset combines free-text CRM notes/call summaries with calibrated stage/score/intent labels under a commercial license.

## What a synthetic-data generation approach would need to cover

Given the above, synthetic generation is the realistic path for the sales/CRM domain specifically. To substitute credibly, it should cover:

**Label schema** (mirroring the three question types):
- `choice`: deal_stage (Prospecting / Qualification / Proposal / Negotiation / Closed-Won / Closed-Lost — richer than the 4-stage Maven set), lead_source category, industry/vertical, buyer_persona/role.
- `score`: lead priority/fit tier (e.g. Cold/Warm/Hot or 1–5), account tier/ICP fit, urgency/timeline tier, budget-fit tier — ordinal, not binary, and should be generated with genuinely graded evidence in the state text (not just a label slapped on).
- `noul`: calibrated yes/no statements like "this lead will respond to outreach," "this deal will close this quarter," "the buyer has budget authority," "there is a competing vendor in this deal," "the stated pain point is a top-3 priority for the buyer" — each needs P(true) grounded in specific, sometimes conflicting evidence in the state so the model has to weigh evidence rather than pattern-match a keyword.

**Input/state variety** needed for calibration to be meaningful (not just accurate):
- Mixed source types: structured CRM fields (amount, stage, dates, owner) *and* free text (call notes, email threads, discovery-call summaries) *and* combinations of both, since real states are heterogeneous.
- Deliberately ambiguous/conflicting states (e.g. high engagement but stated no-budget; senior title but low usage signal) so P(true) targets aren't near 0/1 — this is the part public datasets structurally lack, since real CRM data is mostly clear-cut historical outcomes, not evidence-weighing scenarios.
- Varying evidence strength/density (a two-line note vs. a full transcript) so the model learns to calibrate uncertainty down when evidence is thin, not just to memorize surface cues.
- Multiple verticals/deal sizes/sales motions (SMB self-serve, mid-market, enterprise/multi-stakeholder) so the model doesn't overfit to one company archetype the way every public tabular dataset (X Education, Maven's hardware reseller) does.
- Adversarial/near-miss examples for `choice` and `score` (states that plausibly could be two adjacent stages or tiers) to force genuine calibration rather than confident-but-wrong classification.
- A held-out human-labeled slice (even a few hundred real, anonymized examples if AccuKnox's own CRM data can be used with permission) to sanity-check that synthetic-trained calibration transfers to real inputs — synthetic-only training risks the model calibrating well to its own generator's tells rather than real ambiguity.

# Candidate training data, by domain — verified (Phase 0 complete)

Replaces the earlier unverified starting list. Every dataset below was confirmed to
exist via live web search/fetch by a dedicated research pass per domain, not recalled
from general knowledge. Full detail (label schema, size, quality issues) lives in the
per-domain files linked below — this is the cross-domain summary and the licensing
picture that matters for build decisions.

## 1. Customer support triage — [datasets-customer-support.md](datasets-customer-support.md)
- **Commercially clear**: CFPB Consumer Complaint Database (public domain, millions of rows, clean `Timely response Y/N` noul target, finance-only, no urgency score); Bitext customer-support dataset (CDLA-Sharing-1.0, 26.8K rows, ready-made intent labels, but synthetic).
- **Best schema, wrong license**: Tobi-Bueck multilingual tickets (61.8K rows, 52-way department + 5-level priority) is CC-BY-NC-4.0 — eval-only, not trainable commercially.
- **Gap**: no single commercially-clear dataset combines real tickets + department + urgency in one place. Plan to blend CFPB (real, clear license) with Bitext (labels) and validate against the NC dataset without training on it.

## 2. Trust & safety / content moderation — [datasets-trust-safety.md](datasets-trust-safety.md)
- **Strongest domain found**: three commercially-clear, well-labeled datasets — Jigsaw Civil Comments (CC0, 2M rows, continuous 0-1 score — best `score` fit), Jigsaw Toxic Comment Classification (CC0, 223K rows, multi-label `noul`/`choice`), UC Berkeley Measuring Hate Speech (CC-BY-4.0, 136K annotations, IRT-derived severity).
- Hold back HateXplain, Enron-Spam, OLID — license inconsistent or unconfirmed.

## 3. Email / communication classification — [datasets-email-communication.md](datasets-email-communication.md)
- **Cleanest license**: SpamAssassin Public Corpus (CC0, 6K messages, includes a hard-ham tier useful for calibration).
- **Best schema**: Multilingual Customer Support Tickets (CC BY 4.0, 28.6K rows, native priority + queue).
- **Phishing**: combined Kaggle set (merges CEAS+Nazario+Nigerian Fraud+SpamAssassin) is the most practical binary set; license needs manual Kaggle-page confirmation.
- **Caveat**: none of these reflect current 2025-2026 phishing tactics — dataset staleness is a real gap here, not just a licensing one.

## 4. Sales / CRM — [datasets-sales-crm.md](datasets-sales-crm.md)
- **Tabular lead/deal data is NOT thin**: Maven Analytics CRM Sales Opportunities (~8.8K deals, deal-stage labels), Kaggle Lead Scoring/X Education (9,240 rows, binary + ordinal quality), UCI Online Shoppers Intention (CC BY 4.0, 12,330 sessions) — good `choice`/`noul` sources, no free text.
- **Text-grounded conversation data IS thin and licensing-restricted**: the well-labeled options (CRMArena-Pro, CallCenterEN, a Japanese willingness-aware sales corpus) are all CC-BY-NC research-only. The one commercial-license option (Apache 2.0, 100K rows) is fully synthetic.
- **Confirmed gap**: no dataset anywhere combines free-text CRM notes with calibrated stage/intent labels under a commercial license. **This domain needs synthetic data generation from day one** — the file has a spec for label schema and required input variety (mixed structured+text states, ambiguous evidence, adversarial near-misses).

## 5. Security operations / alert triage — [datasets-security-ops.md](datasets-security-ops.md)
- **Best false-positive/triage fit**: Microsoft GUIDE (Kaggle, CDLA-Permissive-2.0, 1.6M alerts from 6,100+ real orgs, analyst TP/BP/FP labels) — exactly our `noul` use case, though mostly structured columns and one third-party report flagged possible label leakage worth auditing before we rely on it.
- **Best severity fit**: NVD CVE/CVSS (public domain) + CIRCL's packaged HF versions (CC-BY-4.0, 565-779K rows) — terse technical text, not narrative.
- **Best category fit**: VERIS Community Database (CC-BY-SA 4.0, ~9,800 real coded breaches).
- **Confirmed**: CICIDS2017/UNSW-NB15 are real but numeric network-flow features, not usable text for this model — correctly ruled out rather than assumed.

## 6. E-commerce — [datasets-ecommerce.md](datasets-ecommerce.md)
- **Safest picks**: Women's E-Commerce Clothing Reviews (CC0, 23.5K rows — rating for `score`, "Recommended IND" for `noul`, department for `choice`, all in one set) and Amazon Fine Food Reviews (CC0, 568K rows).
- **Confirmed avoid**: Yelp Open Dataset explicitly bars commercial use in its terms.
- **Ambiguous**: Amazon Reviews 2018/2023 academic releases — maintainers describe them as "for research purposes," commercial status unclear.
- **Confirmed gap**: no public dataset has genuine return-reason labels; every one found is synthetic. Matches security-ops and sales-CRM in needing generated data.

## Cross-domain takeaways

- **Three domains are in good shape commercially**: trust & safety, e-commerce (partial), security-ops (partial) — real data, real licenses, minimal synthetic need.
- **Three domains have a real, confirmed synthetic-data dependency, not just a hunch**: sales/CRM (conversational text), e-commerce (return-reason only), security-ops (SOC false-positive text is thin outside GUIDE).
- **License hygiene is the recurring risk across every domain** — multiple datasets otherwise perfect on schema (Tobi-Bueck tickets, Yelp, Amazon Reviews) are commercially unusable or ambiguous. Any dataset marked "Uncertain" in the per-domain files needs a manual license check before it touches a training run.
- **Dataset staleness matters for two domains specifically**: email/phishing (tactics move fast, corpora don't) and security-ops (CVE/CWE taxonomies drift). Plan to budget for periodic refresh, not just initial collection.

## Next actions
- [ ] Manually confirm the "Uncertain" licenses flagged in each per-domain file before any training run touches that data
- [ ] Design the synthetic-data generation pipeline for: sales/CRM conversational states, e-commerce return-reasons, security-ops SOC false-positive narratives
- [ ] Audit Microsoft GUIDE for the reported label-leakage issue before relying on it for the security-ops false-positive expert
- [ ] Decide whether to train/validate against the CC-BY-NC datasets (Tobi-Bueck tickets, CRMArena-Pro, CallCenterEN) for eval-only benchmarking, since their license blocks training use but not internal evaluation

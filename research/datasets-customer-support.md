# Real Datasets for Customer Support Triage (System 1 typed-decision model)

Researched via live web search on 2026-09-27. Every dataset below was confirmed to
exist at a real, working URL at time of writing — no dataset names were guessed
from general knowledge. "Verified" means the existence, approximate size, and
license were confirmed from the dataset's own listing page or search results
quoting that page; "Uncertain" fields are called out explicitly rather than
assumed.

---

## 1. Multilingual Customer Support Tickets (Tobias Bueck)

**Status: Verified (exists, license confirmed; commercial-use restricted)**

- **URL (Hugging Face):** https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
- **URL (Kaggle mirror):** https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
- **License:** `cc-by-nc-4.0` (Creative Commons Attribution-NonCommercial 4.0) — **NON-COMMERCIAL. Does not permit commercial model training without separately licensing from the author.** Flag clearly before use.
- **Size:** ~61.8k rows (single train split), English + German.
- **Labels / schema:** `subject`, `body`, `answer` (free text) plus:
  - `type` — 4 categorical values (e.g., Incident/Request/Problem/Change-style ticket type)
  - `queue` — 52 categorical values representing support departments/teams (e.g., Billing, Technical Support, IT, Product Support)
  - `priority` — 5 categorical values (ordinal urgency)
  - `language` — English/German
  - `tag_1`...`tag_6` — hierarchical free tags (211–956 unique values each)
- **Fit assessment:**
  - `queue` (52-way) → excellent `choice` fit for department routing.
  - `priority` (5-level ordinal) → excellent `score` fit for urgency.
  - `type` (4-way) → good `choice` fit for ticket-type classification.
  - No native refund/churn boolean, but `tag_*` fields could be mined for `noul` statements (e.g., "this is a refund request") with some label engineering.
- **Quality notes:** Real support emails (not obviously synthetic body text), moderate size, bilingual (mostly English/German — limited language diversity beyond that). The **NC license is the main blocker** for a commercial product; would need a paid/alternate license from the author or use for eval-only, not training a shipped commercial model.

---

## 2. Customer Support Ticket Dataset (suraj520, Kaggle)

**Status: Verified (exists); license Uncertain — must be checked on the page directly before use**

- **URL:** https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
- **License:** Not confirmed via search — Kaggle listing did not surface an explicit license string in available results. **Uncertain; verify the "License" field on the dataset page before any use**, since Kaggle datasets range from CC0 to "Other/unspecified."
- **Size:** ~8.4k tickets (commonly cited as "8k+ tech-product support tickets").
- **Labels / schema:** Ticket Type, Ticket Subject, Ticket Description, Ticket Status, **Ticket Priority** (Critical/High/Medium/Low), **Ticket Channel** (Email/Chat/Social media/Phone), Customer Satisfaction Rating, First Response Time, Product Purchased.
- **Fit assessment:**
  - `Ticket Priority` (4-level) → good `score` fit for urgency.
  - `Ticket Type` / `Ticket Channel` → good `choice` fit.
  - `Customer Satisfaction Rating` (numeric, likely 1-5) → could be binarized into a `noul` ("customer is satisfied") or used as an ordinal `score`.
  - No explicit churn/refund flag, but "Ticket Type" often includes a Refund category in these Kaggle-style synthetic ticket sets — verify exact category list on download.
- **Quality notes:** Widely believed to be **synthetic/templated** (common pattern among these small Kaggle "8k tickets" sets, similar to `mirzayasirabdullah07/customer-support-tickets-dataset-200k-records` and `ajverse/customer-support-tickets-crm-dataset`, which appear to be generated from the same template family). English-only, small. Good for bootstrapping/prototyping the `score`/`choice` heads, weak for real-world calibration.

---

## 3. Twitter US Airline Sentiment (Crowdflower / Figure Eight)

**Status: Verified (well-established, widely cited dataset); commercial use restricted**

- **URL:** https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
- **License:** Creative Commons **Attribution-NonCommercial-ShareAlike 4.0** (CC BY-NC-SA 4.0) — **NON-COMMERCIAL.** Not usable for commercial model training without a separate license.
- **Size:** 14,640 tweets from ~7,700 users, collected February 2015, across 6 US airlines (United, US Airways, American, Southwest, Delta, Virgin America).
- **Labels / schema:** `airline_sentiment` (positive/neutral/negative), `airline_sentiment_confidence`, `negativereason` (~10 categories: e.g., "Late Flight," "Customer Service Issue," "Lost Luggage," "Cancelled Flight," "Damaged Goods," etc.) with `negativereason_confidence`.
- **Fit assessment:**
  - `airline_sentiment` (3-way) → excellent `choice` fit for sentiment classification, and a strong template/proxy for building a "customer is upset" `noul` question via calibrated confidence scores (dataset even ships crowd-worker confidence, useful for calibration targets).
  - `negativereason` (10-way, negative-tweets only) → good `choice` fit for a complaint-category / churn-risk-signal classifier.
  - No explicit urgency/priority score or department label.
- **Quality notes:** Old (2015), airline-industry-specific (narrower domain than general support triage), but real crowd-labeled data with confidence scores — a good calibration benchmark despite the license blocking commercial training use as-is.

---

## 4. Bitext Customer Support LLM Chatbot Training Dataset

**Status: Verified (exists, license confirmed); synthetic data, commercial use permitted under CDLA-Sharing**

- **URL:** https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- **License:** **CDLA-Sharing-1.0** (Community Data License Agreement – Sharing, v1.0). This license does permit commercial use, but is share-alike for the *data* — if you publish a modified/derived dataset built from it, that derivative must also be shared under CDLA-Sharing. (It does not automatically require sharing a trained model's weights, but consult counsel on how "derived database" is interpreted for your pipeline.) Flag as "commercial use permitted with share-alike data obligation," not unrestricted.
- **Size:** 26,872 instruction/response rows, ~19.2 MB, ~3.57M tokens.
- **Labels / schema:** `flags` (linguistic-variation tags), `instruction` (customer utterance), `category` (10 high-level categories: ACCOUNT, CANCELLATION_FEE, DELIVERY, FEEDBACK, INVOICE, NEWSLETTER, ORDER, PAYMENT, REFUND, SHIPPING_ADDRESS), `intent` (27 fine-grained intents, e.g., `cancel_order`, `track_refund`, `create_account`, `complaint`), `response`.
- **Fit assessment:**
  - `category` (10-way) → good `choice` fit for department/topic routing (REFUND, PAYMENT, DELIVERY map directly onto real support queues).
  - `intent` (27-way) → good `choice` fit for finer intent classification, and several intents (e.g., `complaint`, `track_refund`, `cancel_order`) are directly usable as positive examples for `noul` questions like "this is a refund request" or "customer wants to cancel."
  - No native urgency/priority `score` or churn-risk field — would need to derive/weak-label these from `intent`.
- **Quality notes:** Explicitly a **hybrid synthetic dataset** ("generated using NLP/NLG technology and automated Data Labeling tools") — not real customer conversations, so tone/style may not match real tickets, but labels are clean and consistent, and it's a full public sibling family (telco, insurance, retail-banking, events-ticketing verticals also exist under the same bitext org on Hugging Face if closer-to-domain data is needed later).

---

## 5. CFPB Consumer Complaint Database

**Status: Verified (official US government dataset, actively maintained)**

- **URL:** https://www.consumerfinance.gov/data-research/consumer-complaints/ (bulk download + Open Data API); mirrored on Kaggle (e.g., https://www.kaggle.com/datasets/selener/consumer-complaint-database) and Google BigQuery public datasets.
- **License:** US government public data — explicitly "freely available for anyone to use, analyze, and build on" (public domain / no restriction on commercial use). This is the **strongest license of the candidates** for commercial training.
- **Size:** Cumulative since 2011, several million complaints (CFPB received ~3.2M complaints in calendar year 2024 alone, and the database goes back to 2011 with narratives added since 2017 for consenting consumers) — total corpus is in the multi-million-row range; complaints with a free-text "consumer complaint narrative" (opt-in) are a smaller but still large subset (commonly cited as 300k-1M+ narratives in various snapshot mirrors).
- **Labels / schema:** `Product` / `Sub-product` (financial product taxonomy — e.g., "Credit card," "Mortgage," "Debt collection," "Checking or savings account"), `Issue` / `Sub-issue` (fine-grained complaint reason taxonomy), `Company response to consumer`, `Timely response?` (Yes/No), `Consumer disputed?` (legacy field), `Consumer complaint narrative` (free text, opt-in subset only), `Company`, `State`, `Date received`.
- **Fit assessment:**
  - `Product`/`Sub-product` → excellent `choice` fit for department/category routing (finance-domain specific, but the taxonomy structure — product then sub-product then issue then sub-issue — is a strong template for a general support-routing schema).
  - `Timely response?` (Yes/No) → direct, real-world `noul` fit ("this complaint received a timely response") with an actual base rate for calibration.
  - `Company response to consumer` (categorical: e.g., "Closed with explanation," "Closed with monetary relief," "In progress") → usable as a `choice` or as a proxy `noul` for "consumer received relief."
  - No explicit urgency/priority score — would need to be derived.
- **Quality notes:** Real, large-scale, continuously updated, high credibility. **Domain-specific to consumer finance** (banking/lending/credit), not general multi-industry support — best used for the churn-risk/dissatisfaction and routing-taxonomy patterns rather than as an exact drop-in for a generic "support ticket" schema. Only the opt-in narrative subset has free text; the rest is structured metadata only.

---

## 6. Customer Support on Twitter (thoughtvector)

**Status: Verified (exists); label-free raw corpus; commercial terms Uncertain**

- **URL:** https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- **License:** Listed license terms are ambiguous in available search results, and the dataset's own description reportedly states that **commercial applications / use of the full dataset require contacting the maintainer directly (stuart@thoughtvector.io)**. Treat as **non-commercial-by-default / requires explicit clearance** — do not assume free commercial use.
- **Size:** ~2.8-3.0 million tweets and replies across major brands (customer tweets + brand-support replies), collected 2017.
- **Labels / schema:** Raw tweet pairs/threads only — `tweet_id`, `author_id`, `text`, `created_at`, `response_tweet_id`, `in_response_to_tweet_id`, `inbound` (flag for customer vs. brand). **No native department/priority/sentiment/churn labels** — this is a large realistic-language corpus, not a labeled classification dataset.
- **Fit assessment:** Low direct fit for any of choice/score/noul out of the box — would require a separate labeling/weak-supervision pass (e.g., LLM-assisted labeling) to produce department, urgency, or churn-risk labels. Useful primarily as a source of **realistic input text distribution** (real customer language, real brand variety) to pair with labels transferred from other datasets, not as a labeled dataset itself.
- **Quality notes:** Real, large, multi-brand (so more industry diversity than the airline-only sentiment set), but stale (2017), unlabeled, and commercial-use terms need direct clearance from the maintainer before any commercial training use.

---

## Recommendation: best 2-3 starting points

1. **Tobi-Bueck / tobiasbueck multilingual customer support tickets** (#1) is the single best structural match — it already has `queue` (department, `choice`), `priority` (urgency, `score`), and `type` (`choice`) on ~62k *real* support emails, which is exactly the three-question-type shape this project needs. The blocker is the **CC-BY-NC-4.0 license** — usable now for prototyping/eval and for shaping the label schema, but commercial training needs a separate license from the author (worth reaching out) or restricting this dataset to internal eval only.

2. **CFPB Consumer Complaint Database** (#5) is the best **commercially-clear, real-world** dataset: public domain, millions of real complaints, a genuine multi-level `Product`/`Sub-product`/`Issue`/`Sub-issue` routing taxonomy to model `choice` after, and a real `Timely response?` boolean that is a clean `noul` example with an actual calibration target. Its finance-only domain means it's better used to validate the *routing/taxonomy pattern* and to seed the churn/dissatisfaction `noul` head than as a literal in-domain training set for a general support-ticket product.

3. **Bitext customer-support LLM dataset** (#4) is the best **commercially-usable, ready-to-use-today** labeled set: CDLA-Sharing-1.0 permits commercial use, categories map cleanly onto real support departments (REFUND, PAYMENT, DELIVERY, ACCOUNT...), and intents give a natural path to refund-request and cancellation `noul` questions. Its main weakness — being synthetic/templated rather than real customer text — means it should be blended with a realistic-text source (e.g., #6, once cleared, or the NC-licensed #1 for eval-only) rather than used alone for tone-sensitive calibration.

Together, #1 (for schema/urgency shape, eval), #5 (for commercially-clear real-world routing/noul patterns), and #4 (for commercially-usable ready labels) cover all three question types with at least one dataset each while being explicit about which pieces are commercially safe today versus which require a licensing conversation.

# E-commerce Training Data — Verified Public Datasets

Research pass for the "System 1" typed-decision model (choice / score / noul questions). Every dataset below was
checked against a live source (Hugging Face, Kaggle, SNAP, or the maintainer's own page) via web search/fetch on
2026-09-27. Nothing here is included on the basis of "this probably exists" — candidates I could not confirm are
omitted rather than padded in.

---

## 1. Amazon Reviews 2023 (McAuley Lab, UCSD)

**Status: Verified**
**URL:** https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 (paper/site: https://amazon-reviews-2023.github.io/)
**License:** No standalone dataset license file. The maintainers' position (stated on the older 2018 release page,
same lab/data lineage) is that they do not own the underlying review text — it is Amazon customer content — and
they release the crawl "for research purposes"; they explicitly say they are "not in a position to offer a license."
The GitHub code repo (processing scripts) carries an MIT-style license, which covers the *code*, not the *review
text/metadata itself*. **Commercial training use is a genuine gray area** — treat as research-use data unless you
get separate clearance; do not assume MIT applies to the underlying reviews.
**Size:** ~571.5M reviews across 33 categories (Gift_Cards ~152K up to Clothing_Shoes_and_Jewelry ~66M), spanning
May 1996–Sep 2023. Comes with rich item metadata (title, category, price, features, images).
**Label schema:** `rating` (1.0–5.0 float) → natural **score** fit. `category`/hierarchical product taxonomy →
natural **choice** fit for product categorization. No return-reason or explicit aspect labels — aspect-level
`noul` questions ("is battery life mentioned positively?") would have to be derived/weak-labeled from free text.
**Known issues:** Verified-purchase flag is present but review text is unmoderated web content (spam, incentivized
reviews); extreme class imbalance across categories and rating (skews 5-star); commercial-use ambiguity noted above.

---

## 2. Amazon Review Data 2018 (Jianmo Ni / Julian McAuley, UCSD) — 5-core subsets

**Status: Verified**
**URL:** http://deepyeti.ucsd.edu/jianmo/amazon/index.html (mirror: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
**License:** Same non-commercial/research-only caveat as #1, stated directly on the page: the maintainers don't own
the data and cannot license it; they ask it be used for research. Predates the 2023 release; still widely cited in
papers because of the standard 5-core train/test splits.
**Size:** 233.1M reviews total (1996–2018); 5-core filtered subset (≥5 reviews per user/item) ≈ 75.26M reviews,
29 categories.
**Label schema:** `overall` rating 1–5 → **score**. `category` → **choice**. Includes "also bought/also viewed"
graphs (not directly useful for this model). No return-reason labels.
**Known issues:** Older snapshot (stale for anything post-2018); same commercial-use ambiguity as the 2023 release;
heavier preprocessing needed to dedupe/clean metadata than the 2023 version.

---

## 3. Yelp Open Dataset

**Status: Verified — commercial use NOT permitted**
**URL:** https://www.yelp.com/dataset (terms: https://s3-media0.fl.yelpcdn.com/assets/srv0/engineering_pages/f64cb2d3efcc/assets/vendor/Dataset_User_Agreement.pdf)
**License:** Explicit custom Yelp Dataset Terms of Use — **"solely non-commercial use"** and users are "barred from
using the data in connection with any commercial purpose." Yelp caps its own liability at $50 and can revoke access.
This is a hard blocker for a commercially trained model unless you request separate written permission from Yelp
(the terms mention narrow carve-outs like journalism, not general commercial ML training).
**Size:** ~6.9M reviews, ~150K businesses (many restaurants), star ratings 1–5, check-ins, tips.
**Label schema:** `stars` 1–5 → **score**. Business `categories` (multi-label, e.g. "Restaurants, Mexican") →
**choice**, though multi-label taxonomies need collapsing to single-label for a `choice` question. No aspect or
return-reason labels (return reasons don't apply to a restaurant/services domain anyway).
**Verdict:** Useful as a review-sentiment reference/eval set only if you have the non-commercial exemption; **do
not use for commercial model training** given the explicit restriction. Flagged per the task's own caution.

---

## 4. Amazon Fine Food Reviews (SNAP / Kaggle)

**Status: Verified**
**URL:** https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews (original: https://snap.stanford.edu/data/web-FineFoods.html)
**License:** **CC0 (Public Domain)** on the Kaggle listing — the most permissive of the review datasets found here,
suitable for commercial training.
**Size:** ~568,454 reviews from ~256,059 users on ~74,258 products, spanning Oct 1999–Oct 2012.
**Label schema:** `Score` 1–5 → **score**. `HelpfulnessNumerator/Denominator` → could derive a `noul` question
("is this review considered helpful?"). No product-category or return-reason field — it's food reviews only, so
category diversity is narrow (single vertical).
**Known issues:** Old (pre-2013) and food-only, so it under-represents general e-commerce categories; some
duplicate reviews (same user/text across products) reported in community analyses.

---

## 5. Women's E-Commerce Clothing Reviews (Kaggle, nicapotato)

**Status: Verified**
**URL:** https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
**License:** **CC0-1.0** — safe for commercial use.
**Size:** 23,486 rows, 10 columns.
**Label schema:** `Rating` 1–5 ordinal → **score**. `Recommended IND` (0/1 binary, "would you recommend this
product") → near-perfect natural fit for a **noul** question ("the reviewer recommends this product: P(true)").
`Division Name` / `Department Name` / `Class Name` (categorical, e.g. Tops/Dresses/Intimates) → **choice** fit for
product categorization at a coarser grain. Review `Title` + `Review Text` give the free text needed as model input.
**Known issues:** Small (23K rows) and single-vertical (women's apparel only); `Class Name` has some missing values;
skews toward positive ratings like most retail review data.

---

## 6. Flipkart E-Commerce Dataset (Kaggle, PromptCloud / atharvjairath mirror)

**Status: Verified**
**URL:** https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products (mirrored as
https://www.kaggle.com/datasets/atharvjairath/flipkart-ecommerce-dataset)
**License:** **CC0: Public Domain**.
**Size:** ~20,000 product listings, 15 columns.
**Label schema:** `product_category_tree` (full breadcrumb, e.g. "Clothing >> Women's Clothing >> Lingerie, Sleep &
Swimwear >> Shorts") → strong **choice** fit for product categorization (multi-level, can pick a granularity).
`description`, `product_name`, `brand`, `retail_price`/`discounted_price` give listing text as model input.
**Known issues:** This is a *listing* dataset, not a review dataset — no rating/sentiment field, so it's useful only
for the categorization leg of the domain, not review sentiment or return-reason. No return/refund info.

---

## Return-reason data: does it need to be synthetic?

**Yes — confirmed.** I searched specifically for real, labeled return-reason datasets (Kaggle, HF, academic
releases) and found none with genuine customer-reported reason labels tied to real transactions. What exists is
exclusively **synthetic/simulated**, e.g.:
- "Synthetic E-Commerce Returns Management Dataset" (Kaggle, sowmihari) — explicitly synthetic.
- "Synthetic Dataset for E-Commerce Return Analysis" (Kaggle, sayalikhot21) — explicitly synthetic.
- "Product Return Risk Prediction" / "Product Return Prediction" (Kaggle) — these predict *whether* an item is
  returned from price/rating/behavioral features; they do not carry a labeled *reason* (e.g. "wrong size,"
  "defective," "changed mind").

Real retailers treat return-reason taxonomies as proprietary (they're tied to internal refund/logistics systems),
so nothing with genuine reason labels is publicly released. **Plan on generating synthetic/templated return-reason
training data** (e.g. LLM-generated return requests conditioned on a reason taxonomy) rather than expecting to find
a real public source — and consider weak-labeling review text that mentions returns ("I sent it back because...")
from the Amazon/clothing datasets above as a small supplementary real-data signal, though this will be sparse and
noisy.

---

## Recommendation

Best starting point, in order:

1. **Women's E-Commerce Clothing Reviews (#5)** — CC0, clean, and the only dataset here with a ready-made `noul`
   label (`Recommended IND`) *and* a `score` label (`Rating`) *and* a `choice` label (`Class Name`/`Department`) in
   the same rows. Small, so use it for held-out eval/calibration checks more than bulk pretraining volume.
2. **Amazon Fine Food Reviews (#4)** — CC0, ~568K rows, gives real volume for the `score` (star rating) task with
   zero licensing risk, and its helpfulness votes are a free secondary `noul` signal.
3. **Flipkart E-Commerce Dataset (#6)** — CC0, best product-categorization (`choice`) source with a genuine
   multi-level taxonomy, complementing the two review datasets which lack listing-only category structure.

Treat the Amazon Reviews 2018/2023 UCSD releases (#1, #2) as a volume source only if the commercial-use ambiguity is
acceptable to the project (or after seeking clarification from the maintainers) — they dwarf everything else in
scale (hundreds of millions of rows across every retail category) and are the only sources here with enough breadth
to cover categorization + sentiment jointly at scale. **Do not use the Yelp Open Dataset (#3) for commercial model
training** — its terms explicitly prohibit commercial use. Return-reason data must be synthesized; no verified real
public dataset with genuine reason labels exists.

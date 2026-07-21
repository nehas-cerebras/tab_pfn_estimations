# TabPFN & TabICL: A Timeline of Tabular Foundation Models

*Compiled 2026-07-09 via multi-source deep research (23 sources fetched, 100 claims extracted, 25 adversarially verified).*

## Ground rules / definitions used throughout

- **Table**: rows (individual examples — e.g. patients, houses, customers) and columns (measurements about each example — e.g. age, price, income). One column is usually the **target** — the thing you're trying to predict.
- **In-context learning**: instead of *learning a formula* from your specific table (what normal machine learning does), you hand the model your whole table — rows where you already know the answer, plus new rows where you don't — in one shot, and it reads all of it at once and guesses the missing answers, the way you'd guess a pattern from a handful of solved examples sitting next to the new problem.
- **Pretrained on synthetic data**: the model is never trained on real tables at all. It's trained once, offline, on millions of *fake, randomly generated* tables that follow made-up statistical rules, so it learns the general skill of "how to read a table and guess a pattern" rather than memorizing any specific real dataset.
- **Transformer / "cross-referencing"**: a network that takes a list of vectors (lists of numbers) as input and, for each one, produces an updated vector by letting each item compute a weighted average of every other item — weights the network decides itself based on how relevant two items look to each other. That's the only mechanism needed to follow everything below; I call it "cross-referencing" throughout.

---

## The timeline

### 1. TabPFN v1 — July 5, 2022 (arXiv:2207.01848)
**Authors**: Hollmann, Müller, Eggensperger, Hutter (University of Freiburg) · Published ICLR 2023

The paper that started the field. Turns table-reading into in-context learning: feed the network your known rows and unknown rows together, it cross-references every row against every other row and every column against every other column, then reads off a guess for each unknown row — all in one pass, no per-dataset training step.

Two big limits: **classification only** (predicting a category, e.g. "spam or not spam" — not a number), and a **fixed maximum table size** (roughly 1,000 rows, 100 columns) — tables with fewer columns than the max had to be padded out with zeros to fit the fixed-size slot the network expected.

### 2. TabPFN v2 — online January 8, 2025, *Nature* Vol. 637, pp. 319–326 (DOI 10.1038/s41586-024-08328-6)
**Authors**: Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister, Hutter

Explicitly framed as fixing v1's "many limitations that rendered it inapplicable in most cases." Builds directly on v1's core idea (read the whole table at once, no training loop):

- Adds **regression** (predicting a number, e.g. a price) — not just categories.
- Scales ~50x: up to **10,000 rows and 500 columns**.
- Handles missing values and categorical columns (e.g. "city name") natively; far more robust to noisy/irrelevant columns.
- Replaces zero-padding with a **learned per-column tokenizer**: each column gets its own small learned vector, with a bit of random noise added, so the model can distinguish columns from each other even in tables whose shape it's never seen before.

### 3. TabICL v1 — submitted February 8, 2025, revised May 24, 2025 (arXiv:2502.05564)
**Authors**: Qu, Holzmüller, Varoquaux, Le Morvan (Inria SODA team) · ICML 2025

A different team enters, and says so directly: TabPFNv2 "largely improves TabPFN in terms of both prediction performance and scalability," but its cross-referencing design — alternating between cross-referencing down columns and across rows, on the *full* table every time — gets too expensive as tables grow. Their fix, building explicitly on TabPFN's approach:

- **Two stages instead of one.** First, compress each row (with all its column values) into a single short summary vector, via cross-referencing across that row's columns and then across other rows. Then pass only those *summary vectors* — one per row, not the whole raw table — into a second, lighter cross-referencing stage that does the actual prediction. Like reading each row, jotting a one-line note about it, then reasoning over the notes instead of re-reading the raw table each time.
- Pretrained on fake tables up to 60,000 rows; usable at prediction time on tables up to **500,000 rows and 500 columns** — 1.5x to 10x faster than TabPFNv2 depending on size, matching its accuracy on smaller tables and beating both TabPFNv2 and the tree-based method CatBoost on the largest tables tested.
- Still classification only, like TabPFN v1.

### 4. TabICL v1.1 — May 2025 (unpublished checkpoint, no paper)

A quiet update: the same TabICL v1 model, re-trained further on an early draft of what would become v2's training data. Still classification-only — no new paper, just a better-trained version of the same architecture.

### 5. TabPFN-2.5 — submitted November 11, 2025, revised February 5, 2026 (arXiv:2511.08667)
**Authors**: Grinsztajn, Flöge, Key, et al., including Hollmann and Hutter

TabPFN's answer to the scale problem TabICL had been pointing at. Billed as "the next generation," pushes the table-size ceiling to **50,000 rows and 2,000 columns** — about 20x more total table cells than TabPFNv2 could handle. It also directly benchmarks against TabICL (v1) as a comparison baseline in its own experiments — the first documented point of cross-team engagement, a full generation before TabPFN-3 went on to adopt TabICL's architecture outright. At this stage the borrowing is still just benchmarking, though: its own feature-grouping change carries no external citation, i.e. it reads as independently developed.

> Caveat: the paper's claim of thereby leading a particular leaderboard (TabArena) did **not** hold up under verification and is omitted here as unconfirmed. The 20x scale-up figure itself is solid.

### 6. TabICLv2 — submitted February 11, 2026 (arXiv:2602.11139)
**Authors**: Qu, Holzmüller, Varoquaux, Le Morvan (same team) · ICML 2026

Builds on TabICL v1's two-stage design, and explicitly benchmarks against TabPFN's newest release (TabPFN-2.5), claiming up to 10x faster prediction on a 50,000-row table on the same hardware. Five concrete changes:

1. **Better fake training data** — an improved generator adding five new families of random rules for generating fake columns and their relationships, on top of TabICL v1's original recipe.
2. **A "scalable softmax"** (called QASSMax) — a tweak to how the cross-referencing step turns its weights into a proper weighted average (softmax = the standard way of turning a list of numbers into weights that sum to 1), so the model generalizes to bigger real-world tables without needing to be pretrained on giant fake tables. Later **adopted directly by TabPFN-3**.
3. **A different training optimizer**, Muon, replacing the standard AdamW optimizer used when fitting the network's numbers during pretraining.
4. First TabICL version to add **regression**, catching up to a capability TabPFN v2 had added a year earlier — via a different mechanism (999 directly predicted quantiles), not something TabPFN-3 later copied.
5. **"Repeated feature grouping,"** an explicitly original grouping scheme (placing each feature into multiple overlapping groups via circular shifts) that the paper itself contrasts against the simpler grouping used by "TabPFNv2 and TabPFN-2.5." Later **adopted directly by TabPFN-3** — a case of TabICL improving on a TabPFN idea, which TabPFN then adopted back.

Its own "Limitations" section admits two real gaps versus TabPFN: missing values are only mean-imputed (not natively handled), and it "does not natively leverage semantic information from column names or textual features" (no text-column support). It has used FlashAttention for efficiency since TabICL v1 (Feb 2025) — earlier than TabPFN's lineage adopted the same external tool (TabPFN-2.5, Nov 2025).

### 7. TabPFN-3 — submitted May 13, 2026, revised May 28, 2026 (arXiv:2605.13986)
**Authors**: Grinsztajn (lead) et al., 41-author Prior Labs technical report including Hollmann, Müller, Schölkopf, LeCun, Hutter

Another big scale jump: up to **1,000,000 rows and 200 columns**, on a single GPU. Its architecture is a significant break from TabPFN v2: rather than keeping v2's cell-level attention design, **the paper's own Figure 5 caption states its architecture is "adapted from the TabICLv2 architecture."** Concretely, it uses TabICL's two-stage row-compression scheme (compress each row into one summary vector, then run in-context learning over those summaries), adds back a reasoning stage the paper describes as "a return to TabPFN v1's" design, and explicitly borrows TabICLv2's feature-grouping scheme and its "QASSMax" scalable-softmax fix. On top of that borrowed base, it adds its own new pieces:

- Memory-saving tricks (splitting rows into chunks, shrinking the intermediate results kept in memory, plus FlashAttention-3 and torch.compile) that let 1,000,000 rows fit on a single GPU. Reported up to 20x faster than TabPFN-2.5, and up to 120x faster specifically for computing SHAP values (a standard way of explaining which columns drove a given prediction).
- New label-encoding and output tricks (spreading label vectors maximally apart before training; predicting via a weighted blend of the most similar training rows' labels, supporting any number of categories) — these appear to be genuine TabPFN-3 originals, not borrowed from TabICL.
- Regression uncertainty via a "bar-distribution regression head" (a discretized probability distribution over value buckets, from which quantiles can be decoded), rather than TabICLv2's direct-quantile approach — a piece TabPFN kept as its own.
- A "thinking" variant (TabPFN-3-Plus) that spends extra compute at prediction time on harder cases.
- Extensions to relational data (multiple linked tables, via an automatic table-join) and mixed text-and-table data.
- A dedicated time-series checkpoint, TabPFN-TS-3.
- Continues distillation into compact MLPs/tree ensembles "via the engine introduced with TabPFN-2.5" (confirmed, not new).
- Notably does **not** adopt TabICL's curriculum pretraining (progressively larger tables during training) or its synthetic-data recipe — those pieces of TabICL were left un-adopted even while the core architecture was borrowed wholesale.

It also benchmarks extensively and directly against TabICLv2 throughout its experiments — this is the opposite of an earlier draft of this document, which incorrectly claimed TabPFN-3 doesn't engage with TabICL at all.

---

## Cross-pollination between the two teams

> **Correction**: an earlier version of this section concluded borrowing ran only from TabICL toward TabPFN. That was based on an abstract-only reading of TabPFN-3 and is wrong — see below.

This isn't two teams working in isolation — it's an open, running, genuinely **two-way** back-and-forth:

- **TabICL → TabPFN, at the paradigm level (2025).** TabICL's first paper (2025) opens by crediting TabPFN as the originator of the "read the whole table at once" paradigm, and frames its own two-stage design purely as a fix for TabPFNv2's scaling bottleneck. TabICLv2 again names "TabPFNv2 and TabICL" together as the two models that "dethroned" traditional tree-based methods, and directly benchmarks against TabPFN's latest release each time. TabICL's time-series forecasting spin-off is described by its own authors as "heavily inspired by" TabPFN's own time-series variant, TabPFN-TS.
- **TabICL → TabPFN, at the architecture level (2026) — the bigger finding.** TabPFN-3's own paper states its architecture (Figure 5) is "adapted from the TabICLv2 architecture," citing Qu et al.'s TabICL and TabICLv2 papers directly, and explicitly says it builds on "the two-stage row-compression design introduced by Qu et al." It also explicitly borrows TabICLv2's specific feature-grouping scheme and its QASSMax scalable-softmax mechanism. TabPFN abandoned its own v2-era architecture and rebuilt around TabICL's, adding a TabPFN v1-flavored reasoning stage and its own new pieces on top. It also benchmarks extensively against TabICLv2 throughout its experiments.
- **TabPFN → TabICL.** TabPFN set the original paradigm and the original (different-mechanism) fix for variable-column tables in TabPFN v2. TabICLv2's "repeated feature grouping" is also explicitly presented as an improvement over the simpler grouping used by "TabPFNv2 and TabPFN-2.5" — so TabICL does engage with specific TabPFN mechanisms, just by improving on them rather than copying them outright the way TabPFN-3 later did with TabICL's architecture.
- **Benchmarking preceded architectural borrowing by a generation.** TabPFN-2.5 (Nov 2025) already compares itself against TabICL (v1) — a full release before TabPFN-3 (May 2026) rebuilt its architecture around TabICLv2.
- **The convergence is selective, not total.** Even after adopting TabICLv2's architecture, feature grouping, and QASSMax, TabPFN-3 kept its own synthetic-data generator, its own regression-uncertainty mechanism (bar-distribution vs. TabICLv2's direct quantiles), its own distillation engine, and added new pieces (orthogonal label embeddings, many-class decoder) with no found TabICL connection. It also didn't adopt TabICL's curriculum-pretraining technique.
- **FlashAttention is convergent adoption of an outside tool, not cross-team borrowing.** TabICL has used it since v1 (Feb 2025); TabPFN's lineage adopted it starting at TabPFN-2.5 (Nov 2025) — both teams independently reaching for the same industry-standard technique (Dao et al., 2022), invented by neither.

**Bottom line, corrected**: TabPFN set the initial paradigm; TabICL responded with a genuinely novel architecture built specifically to fix a cost problem in TabPFNv2, and refined specific TabPFN mechanisms (feature grouping) along the way; TabPFN then tracked TabICL's numbers for a generation (TabPFN-2.5) before — rather than simply out-scaling TabICL with its own architecture (the earlier, incorrect conclusion) — the latest TabPFN generation openly adopted TabICL's architecture wholesale, while still keeping several of its own pieces independent. By 2026, the two lineages' architectures are converging, though not merging completely.

---

## Caveats and open questions

- Two claims were explicitly refuted during adversarial verification and excluded from this report:
  1. TabPFN-2.5's 20x scale-up "enabling it to lead the TabArena benchmark" (the scale-up figure is corroborated; the benchmark-leadership claim is not).
  2. A TabPFN v2 author list that included "Eddie Bergman" (the correct 8-author list per *Nature*/Crossref does not include this name).
- Several supporting details carried medium-confidence (2-1) verification votes rather than unanimous: TabPFN-2.5's exact authorship/date framing, TabICL's 60K-pretrain/500K-inference scaling detail, and the zero-padding-vs-tokenizer contrast between TabPFN v1 and v2. The broad direction of each claim is solid; treat precise mechanistic wording with slight caution.
- The field is moving fast: TabICLv2 and TabPFN-3 were both submitted in 2026, within months of "today" (2026-07-09) — newer unpublished or in-review work may already supersede parts of this timeline.
- **Correction**: this document originally stated TabPFN-3 kept TabPFN v2's cell-level attention design and that "TabPFN-3 does not mention TabICL by name." Both statements were wrong — confirmed by checking the actual paper (Figure 5 and its caption), TabPFN-3's architecture is explicitly adapted from TabICLv2, and TabPFN-3 cites and benchmarks against TabICLv2 extensively. A follow-up double-check of every other claim in this document (regression mechanisms, curriculum training, distillation, FlashAttention, missing-value handling, feature grouping) turned up several more corrections, now applied throughout: TabPFN-3's regression uncertainty uses a "bar-distribution" head (not TabICLv2-style quantiles), it does *not* use curriculum pretraining, its distillation and FlashAttention-3/torch.compile use is now confirmed rather than assumed, TabICLv2's missing-value handling is confirmed to be mean-imputation only (not native), TabICLv2 lacks text/relational support (confirmed via its own Limitations section), TabICLv2 has used FlashAttention since v1 (not "not included" as an earlier draft said), and TabICLv2's "repeated feature grouping" is confirmed as its own original contribution that TabPFN-3 later adopted.
- Open questions not resolved by this research pass:
  - Was the architectural convergence at TabPFN-3 abrupt, or are there intermediate signs in unpublished TabPFN-2.5 revisions?
  - How does TabICLv2 perform against TabPFN-3 specifically on TabICLv2's own benchmarks (TabPFN-3's paper reports beating TabICLv2 on TabPFN-3's own large-data benchmark — an independent, symmetric comparison wasn't found)?
  - Does TabICLv2 (or a future TabICL version) reciprocally adopt anything from TabPFN-3's additions (the many-class decoder, orthogonal label embeddings, bar-distribution regression head, or its memory-efficiency tricks)?
  - What optimizer does TabPFN-3 actually use (not disclosed in any source checked, despite two separate attempts)?
  - Exactly what mechanism does TabPFN v2 itself use for missing values and regression (Nature's paywall blocked direct verification both times it was attempted) — TabPFN-3's confirmed mechanisms may or may not match v2's original ones.

## Primary sources

- TabPFN v1: https://arxiv.org/abs/2207.01848
- TabPFN v2: https://www.nature.com/articles/s41586-024-08328-6
- TabICL v1: https://arxiv.org/abs/2502.05564
- TabPFN-2.5: https://arxiv.org/abs/2511.08667
- TabICLv2: https://arxiv.org/abs/2602.11139 · https://github.com/soda-inria/tabicl
- TabPFN-3: https://arxiv.org/abs/2605.13986
- Secondary analysis ("A Closer Look at TabPFN v2"): https://arxiv.org/abs/2502.17361

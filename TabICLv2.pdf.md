## **TabICLv2: A better, faster, scalable, and open tabular foundation model** 

**Jingang Qu**<sup>* 1</sup> **David Holzm¨uller**<sup>* 1</sup> **Ga¨el Varoquaux**<sup>1 2</sup> **Marine Le Morvan**<sup>1</sup> 

### **Abstract** 

Tabular foundation models, such as TabPFNv2 and TabICL, have recently dethroned gradientboosted trees at the top of predictive benchmarks, demonstrating the value of in-context learning for tabular data. We introduce TabICLv2, a new state-of-the-art foundation model for regression and classification built on three pillars: (1) a novel synthetic data generation engine designed for high pretraining diversity; (2) various architectural innovations, including a new scalable softmax in attention improving generalization to larger datasets without prohibitive long-sequence pretraining; and (3) optimized pretraining protocols, notably replacing AdamW with the Muon optimizer. On the TabArena and TALENT benchmarks, TabICLv2 without any tuning surpasses the performance of the current state of the art, RealTabPFN-2.5 (hyperparameter-tuned, ensembled, and fine-tuned on real data). With only moderate pretraining compute, TabICLv2 generalizes effectively to million-scale datasets under 50GB GPU memory while being markedly faster than RealTabPFN-2.5. We provide extensive ablation studies to quantify these contributions and commit to open research by first releasing inference code and model weights at https://github. com/soda-inria/tabicl, with synthetic data engine and pretraining code to follow. 

### **1. Introduction** 

Tabular data, whether stored in spreadsheets or databases, is ubiquitous across applications ranging from healthcare to credit card fraud detection (Borisov et al., 2022; Jesus et al., 2022; Grinsztajn et al., 2025). While supervised learning on tabular data has long been dominated by gradient-boosted decision trees (Grinsztajn et al., 2022), both pretrained and 

> *Equal contribution 1SODA Team, INRIA Saclay, Palaiseau, France 2Probabl, France. Correspondence to: Jingang Qu _<_ jingang.qu@inria.fr _>_ , David Holzmuller¨ _<_ david.holzmuller@inria.fr _>_ , Marine Le Morvan _<_ marine.lemorvan@inria.fr _>_ . 

_Preprint. February 12, 2026._ 



<!-- Start of picture text -->
RandomForest<br>25% Default ExtraTrees<br>Tuned FastaiMLP<br>Tuned + Ens. EBM<br>Pareto Front TorchMLP<br>20% Mitra<br>TabICL<br>LimiX<br>TabPFNv2<br>15% xRFM<br>XGBoost<br>ModernNCA<br>LightGBM<br>10% CatBoost<br>TabM<br>TabDPT<br>RealMLP<br>5% RealTabPFN-2.5<br>TabICLv2<br>10 − 1 10 0 10 1 10 2 10 3 10 4<br>Train time per 1K samples (s) (median)<br>Optimal<br> 100%<br>×<br>�<br>best<br> − err<br>err<br>�<br> avg<br>Improvability =<br><!-- End of picture text -->

_Figure 1._ **Improvability vs. train time on TabArena (Erickson et al., 2025).** Improvability (lower is better) measures the relative error gap to the best method, averaged across datasets. Train time is training + inference in 8-fold cross-validation. For foundation models, it is dominated by forward passes that perform in-context learning. _Default_ uses default hyperparameters; _Tuned_ selects the best of 200 random hyperparameter configurations on validation; _Tuned + Ens._ applies post-hoc weighted ensemble of all configurations. The runtime of TabICLv2 is measured on an H100 GPU, while others are from TabArena. Results for inapplicable modeldataset pairs are imputed with default RandomForest. 

trained-from-scratch deep learning models have recently been able to match or even surpass their accuracy on tables with up to 100K samples (Erickson et al., 2025; Ye et al., 2024). In particular, starting from TabPFN (Hollmann et al., 2022), tabular foundation models (TFMs) have received a lot of attention thanks to their ability to perform training and inference in a single forward pass of a Transformer-based architecture. The development of better TFMs also benefits downstream adaptations, such as causal inference, generative modeling, joint predictive distributions, and simulationbased inference (Ma et al., 2025b; Robertson et al., 2025; Balazadeh et al., 2025; Hollmann et al., 2025; Hassan et al., 2025; Vetter et al., 2025). To foster this research, there is a pressing need for fully open-source TFMs that rival closedsource ones to democratize access to top-tier performance and demystify the recipe behind top-performing TFMs. 

**Contributions** We introduce TabICLv2, a state-of-theart tabular foundation model, as shown in Figure 1. Our contributions include architectural innovations (Section 3), pretraining improvements (Section 4), a novel synthetic data 

1 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

generator (Section 5), extensive evaluations (Section 6), and an ablation study (Section 7). 

### **2. Related Work** 

#### **2.1. Tabular foundation models** 

Tabular foundation models (TFMs) based on Prior-data fitted networks (PFNs, Muller¨ et al. 2021) emerged as a paradigm shift in tabular learning. Given a training set and test input, a TFM _qθ_ with parameters _θ_ directly predicts a distribution _qθ_ ( _y_ test _| x_ test _, D_ train) with a forward pass on input ( _x_ test _, D_ train). For a single dataset, it hence performs in-context learning (ICL) without gradient updates. For pretraining, given a prior _p_ ( _D_ ) from which datasets _D_ (train+test) can be sampled, TFMs are trained to minimize 



**Architectural perspectives.** TabPFN (Hollmann et al., 2022) treats each row as a token and performs ICL over rows. TabPFNv2 (Hollmann et al., 2025) moves to a cell-based design with alternating row and column attentions, where each cell receives a separate representation. However, this incurs _O_ ( _n_<sup>2</sup> _m_ + _nm_<sup>2</sup> ) complexity for a table with _n_ rows and _m_ columns. TabPFN-2.5 (Grinsztajn et al., 2025) extends TabPFNv2 with deeper networks. TabICL (Qu et al., 2025) reduces the computational complexity to _O_ ( _n_<sup>2</sup> + _nm_<sup>2</sup> ) via a two-stage design: a lightweight column-then-row attention first constructs fixed-dimensional row embeddings, after which ICL is performed over these embeddings. Recent work continues to innovate on these foundations, including Mitra (Zhang et al., 2025a), LimiX (Zhang et al., 2025b), and Orion-MSP (Bouadi et al., 2025). 

**Synthetic prior datasets.** Synthetic priors are central to PFN-style TFMs. TabPFN uses structural causal models (SCMs). TabICL and TabForestPFN (Breejen et al., 2024) extend these by mixing tree-based priors to inject tree inductive biases. Mitra studies prior design principles and proposes mixed priors (SCM + tree ensembles) to better control decision boundaries. TabPFNv2 enriches priors with more sophisticated DAG construction and computational mappings. LimiX introduces hierarchical SCMs with controllable difficulty. Drift-Resilient TabPFN (Helli et al., 2024) tackles temporal distribution shifts with a two-level generative prior that modulates SCM parameters over time. However, TabDPT (Ma et al., 2025a) shows that large-scale pretraining on real datasets can be competitive, and RealTabPFN (Garg et al., 2025) indicates that continued pretraining on real datasets can improve TabPFNv2. 

**Fine-tuning, retrieval, and distillation.** Beyond pretraining, adaptation strategies for TFMs include (a) fine-tuning to shift the learned prior toward a target distribution (Feuer 

et al., 2024; Liu & Ye, 2025; Garg et al., 2025; Kolberg et al., 2025; Rubachev et al., 2025), (b) retrieval-based context selection to enhance compute-constrained scalability (Thomas et al., 2024; Xu et al., 2024; Zhang et al., 2025b; Sergazinov & Yin, 2025), and (c) distilling TFMs into compact MLPs or trees (Bonet et al., 2024; Mueller et al., 2024; Grinsztajn et al., 2025). 

**LLM-based tabular models.** In parallel to table-native TFMs, large language models (LLMs) have been adapted to tabular data via table serialization and continued pretraining (Hegselmann et al., 2023; Gardner et al., 2024; Dong et al., 2025), which are promising but underperform TFMs when sufficient training data is available. 

#### **2.2. Attention struggles with long-context generalization** 

**Attention fading.** Attention is central to PFN-style TFMs. But standard attention, based on softmax, suffers from _attention fading_ (Velickoviˇ c et al.´ , 2024; Nakanishi, 2025): the softmax denominator increases as context length _n_ grows, causing attention distributions to flatten and preventing sharp focus on relevant tokens. This limits length generalization, as models trained on shorter sequences cannot maintain discriminative attention patterns when applied to longer ones. 

**Temperature scaling.** To address attention fading, some work focuses on softmax alternatives (Peters et al., 2019; Ramapuram et al., 2024), which, however, requires specialized implementations incompatible with the softmax ecosystem, e.g., FlashAttention (Dao et al., 2022). We thus focus on a less invasive solution: _temperature scaling_ . Standard attention (Vaswani et al., 2017) already incorporates a fixed scaling temperature factor 1 _/√d_ to prevent dot-product magnitudes from growing with dimension, but this does not address length-dependent fading. YaRN (Peng et al., 2023) scales temperature with context length for RoPE (Su et al., 2021) extension, but the scaling is fixed and requires positional encoding. Recent work proposes dynamic alternatives. Scalable Softmax (SSMax, Nakanishi 2025) scales attention logits by _s_ log _n_ , where _s_ is a learnable per-head parameter. Concurrent theoretical analysis (Chen et al., 2025b) establishes that log _n_ scaling is necessary to maintain attention sharpness as context length _n_ grows. Adaptive-Scalable Entmax (ASEntmax, Vasylenko et al. 2025) extends this with content-aware scaling _δ_ + _β_ (log _n_ )<sup>_γ_</sup> , where _δ_ is a lengthindependent constant offset, and _β_ = softplus(MLP( _X_ )) and _γ_ = tanh(MLP( _X_ )) are input-dependent. Selective Attention (Zhang et al., 2024) takes a different approach by introducing query-dependent temperature _τ_ ( _q_ ) via lightweight MLPs, decoupling the scaling from context length entirely. 

2 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

### **3. Architecture** 

The architecture of TabICLv2 is illustrated in Figure 2. Following TabICL, TabICLv2 chains column-wise embedding, row-wise interaction, and dataset-wise ICL, thus preserving the efficiency of TabICL with a runtime complexity of _O_ ( _n_<sup>2</sup> + _nm_<sup>2</sup> ) for tables with _n_ rows and _m_ columns. In addition, we introduce several improvements that significantly enhance performance without increasing model size (see the ablation study in Section 7). In the following, we mark these improvements with ▶. We focus below on architectural innovations. For model configuration details (e.g., number of layers), refer to Appendix A.4. We provide a short self-contained implementation of the TabICLv2 architecture for educational and experimental purposes, inspired by nanoTabPFN (Pfefferle et al., 2025), at 

https://github.com/soda-inria/nanotabicl . 

▶ **Repeated feature grouping.** TabICL embeds each feature independently, which can lead to representation collapse when features share similar distributions. TabPFNv2 and TabPFN-2.5 mitigate this collapse by grouping multiple columns into single tokens, which also reduces the number of effective features to improve efficiency, but this reduction may lose fine-grained feature information. We propose _repeated feature grouping_ , which places each feature into multiple groups via circular shifts while preserving the number of effective features. Specifically, for a table with _m_ columns, we create _m_ groups where the _j_ -th group contains columns at positions ( _j, j_ + 1 _, j_ + 3) mod _m_ . Each group is encoded by a shared linear layer Lin : R<sup>3</sup> _→_ R<sup>_d_</sup> : 



The shift pattern (0 _,_ 1 _,_ 3) ensures that for _≥_ 7 columns, no pair of columns appears together in more than one group. We show in Appendix A.1 that this pattern generalizes to arbitrary group sizes, although we did not observe consistent improvements from larger groups. 

▶ **Target-aware embedding.** We find it beneficial to inject target information early. After repeated feature grouping produces input data representation _E_ 1 _∈_ R<sup>_n×m×d_</sup> , we add target embeddings to each training token: 



where EmbedTAE is a linear layer for regression or a learnable lookup table for classification. Unlike TabPFNv2 appending the target as an additional column, we directly add target embeddings to all features. This also helps mitigate representation collapse because even when two features share similar distributions, their association with target values often differs across samples. 

**Compression then ICL.** TabICLv2 processes _E_ 2 in three stages: (1) _column-wise embedding_ applies a set transformer 



<!-- Start of picture text -->
Columns (Features)<br>Dataset-wise in-context learning<br>Transformer  with QASSMax<br>+ +<br>Circular feature<br>shift by (0, 1, 3)<br>Training samples Test samples<br>Repeated<br>feature<br> grouping Row-wise interaction<br>Transformer<br>Linear(   )<br>Column-wise embedding<br>Transformer  with QASSMax<br> Training  +<br>samples +<br> Test  Target-aware embedding<br>samples (only for training samples)<br>Rows (Samples)<br><!-- End of picture text -->

_Figure 2._ **Architecture of TabICLv2.** Given an input _X ∈_ R<sup>_n×m_</sup> , _repeated feature grouping_ encodes columns into multiple groups via circular shifts to break feature symmetries, and _target-aware embedding_ injects target information from the beginning. TFcol embeds each feature through a set transformer, TFrow aggregates features into row representations _h_ , and TFicl performs in-context learning to predict test targets _y_ ˆ. QASSMax, our query-aware scalable softmax, is applied in the part of TFcol where inducing points aggregate input information and TFicl to mitigate attention fading and improve long-context generalization. 

TFcol (Lee et al., 2019) to each column; (2) _row-wise interaction_ uses a transformer TFrow with [CLS] tokens to collapse feature embeddings per row into a single vector; (3) _datasetwise ICL_ combines row embeddings with target embeddings and uses a transformer TFicl where test samples attend to training samples for prediction. See Appendix A.2 for details. ▶ Compared to TabICL, our key innovation here is applying a novel scalable softmax to TFicl and to the part of TFcol where inducing points aggregate input information. 

▶ **Query-aware scalable softmax.** To improve generalization to larger datasets, we extend Scalable Softmax (SSMax, Nakanishi 2025), a temperature scaling method that sharpens attention distributions by rescaling queries before computing logits. Let _qh_ = ( _qhi_ ) be a query vector at head _h_ with head dimension indexed by _i_ , and let _n_ be the size of the training set. SSMax rescales queries with a learnable per-head scalar _sh_ : 



We propose query-aware scalable softmax (QASSMax), 

3 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
No SSMax SSMax QASSMax No SSMax SSMax QASSMax<br>1 . 0 C2 C2 C2<br>0 . 9 0 . 80 Anchor sample<br>C3 C1 C3 C1 C3 C1 Test samples<br>0 . 8 0 . 75 C i Negative clusters<br>0 . 7 C4 C4 C4<br>0 . 70<br>0 . 6<br>C2 C2 C2<br>0 . 5<br>0 . 65<br>0 . 4 C3 C1 C3 C1 C3 C1<br>0 10000 0 10000 C4 C4 C4<br>Negative samples Negative samples<br>entropy<br>1K negative samples<br>Accuracy attention<br>Norm.<br>15K negative samples<br><!-- End of picture text -->



<!-- Start of picture text -->
(a)  Accuracy and attention entropy  vs.  negative samples (b)  Decision boundaries for anchor and negative clusters<br><!-- End of picture text -->

_Figure 3._ **SSMax variants mitigate attention fading in a synthetic 2D classification task.** We create a dataset consisting of four negative clusters (C1–C4) and one anchor cluster containing a single anchor sample (triangle) in the training set. We increase negative samples while evaluating 20 fixed test samples (red squares) nearest to the anchor. **(a)** Attention entropy is divided by log _n_ to ensure values in (0 _,_ 1) and averaged across all heads and layers in TFicl, measuring how uniformly test samples attend to training ones. Without SSMax, accuracy drops and entropy rises as negative samples increase, which is a hallmark of attention fading where the model fails to focus on the relevant anchor. QASSMax maintains 100% accuracy with consistently low entropy. **(b)** shows decision boundaries at 1K and 15K negative samples. The region of the anchor cluster shrinks for all variants as negative samples increase. No SSMax collapses at 15K, while QASSMax preserves a stable boundary containing all test samples. 

which rescales each query element as: 



where for _H_ attention heads, MLPbase : R _→_ R<sup>_H×d_head</sup> and MLPgate : R<sup>_d_head</sup> _→_ R<sup>_d_head</sup> are two-layer MLPs with 64 hidden neurons and GELU activation. 

We design QASSMax based on the following rationale: (a) The log _n_ factor is critical as it counteracts the linear growth of the softmax denominator with respect to _n_ (Nakanishi, 2025; Chen et al., 2025b); (b) ASEntmax (Vasylenko et al., 2025) uses learnable _δ_ + _β_ (log _n_ )<sup>_γ_</sup> , inspiring us to generalize to MLPbase(log _n_ ); (c) Element-wise scaling increases expressiveness beyond per-head scalars; (d) Selective Attention (Zhang et al., 2024) introduces query-awareness in temperature scaling, which motivates us to use the bounded query-aware gating _∈_ (0 _,_ 2) that modulates the base scaling without dominating the log _n_ trend. In addition, our gating design shares similar insights with Gated Attention (Qiu et al., 2025), which applies gating to attention outputs and finds query-dependent, element-wise gating most effective. 

QASSMax applied to TFcol and TFicl yields substantially performance improvements, as shown in the ablation study (Section 7). To study its effect on attention fading, we design a toy needle-in-haystack classification task (Figure 3): the model must focus on a single anchor sample (the needle) among increasing negative samples (the haystack). Without scalable softmax, attention entropy rises and accuracy drops. However, QASSMax maintains low entropy and 100% accuracy even with 15K negatives, outperforming SSMax which largely degrades at extreme scales. 

**Many-class classification.** Like many TFMs, TabICLv2 is pretrained with up to 10 classes. We use hierarchical classification (Qu et al., 2025) at the ICL stage for more classes. However, target-aware embedding introduces labels before hierarchical partitioning. ▶ To address this, we propose _mixed-radix ensembling_ : for _C >_ 10 classes, we compute balanced bases [ _k_ 0 _, . . . , kD−_ 1] with each _ki ≤_ 10 and<sup>�</sup> _i_<sup>_ki≥C_, then decompose each label</sup><sup>_y_into</sup><sup>_D_digits</sup> _y_<sup>(</sup><sup>_i_)</sup> _∈{_ 0 _, . . . , ki−_ 1 _}_ via mixed-radix representation. Each digit defines a coarser grouping of the original classes. We run TFcol once per digit and average the outputs: 



Combined with hierarchical classification in TFicl, this enables TabICLv2 to handle an arbitrary number of classes. See Appendix A.3 for details. 

▶ **Quantile predictions for regression.** Existing TFMs adopt different strategies for regression: TabPFNv2 and TabPFN-2.5 model the full predictive distribution by discretizing the target space into bins and applying crossentropy loss, while Mitra and TabDPT predict point estimates using MSE loss. In addition, like most TFMs except LimiX, we train separate models for classification and regression. 

TabICLv2 instead predicts 999 quantiles at probability levels _α ∈{_ 0 _._ 001 _,_ 0 _._ 002 _, . . . ,_ 0 _._ 999 _}_ , trained with pinball loss summed across all quantiles. In preliminary experiments using RMSE evaluation, we found that quantile regression outperforms MSE and the bin-based approach of TabPFNv2. 

At inference, for point estimation we simply average the 

4 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

predicted quantiles, which proves both fast and effective. For probabilistic predictions, we construct a full distribution from the quantiles by enforcing monotonicity via sorting (the default) or isotonic regression (Barlow & Brunk, 1972; Busing, 2022), extrapolating tails with parametric exponential models, and deriving closed-form PDF, CDF, and moments. See Appendix I for details. 

### **4. Pretraining and Inference** 

#### **4.1. Pretraining setup** 

We significantly improve the pretraining setup compared to TabICL, the reference TFM with open pretraining. 

**Three pretraining stages.** We retain the three-stage structure of TabICL that progressively expands the size of pretraining datasets, with up to 100 features throughout all stages. However, following TabPFNv2, we reduce the batch size to 64, allowing more steps with fewer datasets ( _≈_ 35M) than TabICL ( _≈_ 83M) and TabPFNv2 ( _≈_ 130M). The three stages are: 

- **Stage 1** : 500K steps on datasets with 1,024 samples, 30–90% for training, max learning rate 8e-4. 

- **Stage 2** : 40K steps on datasets with 400–10,240 samples (log-uniform), 80% for training, max learning rate 1e-4. 

- **Stage 3** : 10K steps on datasets with 400–60K samples (log-uniform), 80% for training, max learning rate 2e-5. 

In Appendix B.1, we show that stages 2 and 3 yield progressive performance improvements, especially on large datasets. 

**Optimizer.** We use the Muon optimizer (Jordan et al., 2024b) based on the implementation of Schaipp (2025) instead of AdamW (resp. Adam) used by TabICL (resp. TabPFNv2). Following Moonlight (Liu et al., 2025), this implementation multiplies the learning rate for each parameter _W ∈_ R<sup>_n×m_</sup> by 0 _._ 2 _·_ ~~�~~ max _{n, m}_ . We find higher learning rates preferable for Muon. As a result, we use a max learning rate of 8e-4 for the stage 1 compared to 1e-4 for AdamW in TabICL. We adopt cautious weight decay (Chen et al., 2025a) with parameter 0 _._ 01, which applies decay only when the update and parameter have the same sign, avoiding interference with beneficial gradient directions. We also increase gradient clipping from 1 to 10 for stages 1 and 2, sample different train/test sizes per micro-batch, and use a cosine learning rate schedule across all stages. 

**Pretraining cost.** On H100 GPUs with 80GB memory, stage 1 takes around 20 GPU-days, stage 2 around 2.5 GPUdays, and stage 3 around 2 GPU-days, totaling 24.5 GPUdays per model. Given that one H100-hour is roughly equiv- 

alent to 2 A100-hours, our pretraining cost is lower than TabICL (60 A100-days). 

#### **4.2. Inference optimizations** 

We implement _disk offloading_ (Appendix H.2), reducing requirements to under 24 GB CPU and 50 GB GPU to process a table with 1M samples and 500 features within 450 seconds (Figure H.2). Combined with QASSMax for long-context generalization, TabICLv2 can natively handle million-scale tables without retrieval and distillation. In addition, we reduce redundant computation by selectively computing _Q/K/V_ projections (Appendix H.1). 

### **5. Synthetic data prior** 

Our pretraining data is _entirely synthetic_ , following the approach pioneered in TabPFN (Hollmann et al., 2022). The data-generating mechanism is termed _prior_ , as it implicitly defines a Bayesian prior over datasets. For TabICLv2, we design a new prior that retains the structural causal model framework used in Hollmann et al. (2022), incorporates innovations brought by TabICL and TabPFNv2 priors, and adds many novel design options and sampling mechanisms (see Appendix E.1). Unlike architectural and pretraining choices, the new prior is developed mostly without experimental feedback, since fine-grained ablations are impractical and prone to overfitting validation datasets. Instead, the prior development is guided by general design principles (Wilson & Izmailov, 2020), maximizing dataset diversity (e.g., variable dependencies and categorical cardinalities) while encoding useful inductive biases and preserving computational efficiency. This new prior is key to the final performance: pretraining TabICLv2 with the TabICL prior yields substantially lower performance (Figure 10, gray). An ablation using the TabPFNv2 prior is not possible, as it is not open-source. We provide a high-level prior description below and defer details to Appendices E and F. 

**High-level structure.** Figure 4 summarizes the TabICLv2 prior. We first sample global dataset properties, such as the number of numerical and categorical features, and the dataset size. We then sample a directed acyclic graph and random functions defining parent–child relationships, yielding a causal data-generating model. To obtain a dataset with _n_ samples, a matrix _X ∈_ R<sup>_n×di_</sup> of _n_ random vectors is sampled at each root node _i_ and propagated through the graph. Each dataset feature is extracted from a randomly assigned node. Only a subset of a node’s dimensions is used to generate each feature, leaving other dimensions unobserved and thereby introducing noise into the dataset. Unlike prior work, we do not add Gaussian noise at the node level. For numerical features (e.g., _x_ 1), feature values are extracted from a single node dimension. For categorical features (e.g., 

5 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
a) Random graph b) Propagate random vectors c) Computation inside a node d) Random functions e) Random datasets<br>MLP Tree ensemble<br>apply random function<br>standardize + Discretize GP<br>feature scaling<br>conversion Linear Quadratic<br>random rescale EM Product<br>no yes<br>Dataset good?<br><!-- End of picture text -->

_Figure 4._ **High-level structure of the synthetic dataset generation prior.** Random vectors (one per sample) are propagated through a randomly generated graph where each node computes a random function of its parents. Columns of the final dataset are extracted from randomly assigned nodes. The resulting datasets can be rejected based on different filtering criteria. (d) List of the 8 random functions applied: (MLP) Multilayer perceptrons, (Tree Ensemble) Ensembles of symmetric trees inspired by CatBoost (Prokhorenkova et al., 2018), (Discretize) Discretization to nearest neighbors among a random set; (GP) Multivariate Gaussian process functions; (Linear) Linear functions; (Quadratic) Multivariate quadratic functions; (EM) functions with plateaus inspired by the cluster assignment in the EM algorithm; (Product) products of other random functions. (e) Examples of generated 2D classification datasets (cf. Figure F.1). 

_x_ 2), multiple node dimensions are extracted and discretized, either via nearest-neighbor assignment or by applying a softmax to obtain a categorical distribution. 

**New sampling mechanisms** Since the **random graph** sampling mechanism used by Hollmann et al. (2025) can only generate tree-structured graphs, we introduce a “random Cauchy graph” mechanism, which models different global and local node connectivities and is described in Appendix E.4. 

The relation between a child node and its parents is generated using several steps depicted in Figure 4(c). The key step consists in sampling diverse **random functions** to apply to the parent data. We use eight types of random functions, listed in Figure 4(d). The first three are adapted from TabPFNv2 while the other five are new. These functions are chosen to cover different levels of smoothness (which we prove for Gaussian Process functions) and different types of inductive biases (e.g., plateaus or axis-alignment). To handle the case of more than one parent node, we randomly select between two options: concatenate all parent matrices and apply a single random function, or apply random functions to every parent matrix and aggregate the results using sum, product, max, or logsumexp. Even within each function type, we diversify the generated functions using new or extended building blocks including multiple random matrix types (for MLP, linear, quadratic functions, etc), random weight vectors (for singular values, feature importances, etc.), and random activations (for MLPs, random matrices), see Appendix E. 

After applying the random function, we standardize _X_ and 

randomly rescale its columns to emulate different feature importances. The random converters extract feature values but can also modify node values, applying a warping function to scalars or discretization mechanisms to sub-vectors (see Appendix E.6). Finally, the node data _X_ is multiplied by a random scalar emulating a “node importance”. 

**Postprocessing.** We apply some postprocessing similar to TabICL (Qu et al., 2025), including discarding problematic columns and datasets, permuting columns and class labels, and preprocessing features and targets. 

**Data filtering.** Inspired by Dong et al. (2025) and Zhang et al. (2025b), we filter out datasets on which a simple ExtraTrees model cannot improve on a constant baseline according to a bootstrap test. In addition, we directly filter graphs in which nodes associated to _x_ do not have common ancestors with the node associated to _y_ , which implies that _y_ is independent of _x_ . In pretraining stage 1, roughly 35% classification and 25% regression datasets are filtered. Figure 10 shows that filtering improves the convergence of pretraining. 

**Sampling correlated scalars.** We often sample numerical or categorical scalars (“hyperparameters”) from the same distribution multiple times, e.g., the number of categories within a column. We introduce a correlated way to sample them, e.g., to make it more likely to sample datasets where many columns have the same number of categories. 

6 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
MLP<br>Pareto Front ExcelFormer<br>ResNet<br>25% RandomForest<br>AutoInt<br>MLP-PLR<br>FT-Transformer<br>20% XGBoost<br>LightGBM<br>CatBoost<br>TabR<br>15% RealMLP<br>ModernNCA<br>TabICL<br>TabPFNv2<br>10% LimiX<br>TabPFN-2.5<br>RealTabPFN-2.5<br>TabICLv2<br>10 0 10 1<br>Inference time per 1K samples (s) (median)<br>Optimal<br> 100%<br>×<br>�<br>best<br> − err<br>err<br>�<br> avg<br>Improvability =<br><!-- End of picture text -->

_Figure 5._ **Improvability vs. inference time on TALENT (Ye et al., 2024).** The runtime of TabICLv2 is measured on an H100 GPU, while other runtimes are taken from TALENT. 

### **6. Experiments** 

**Benchmarks.** We use the TabArena (Erickson et al., 2025) and TALENT (Ye et al., 2024) benchmarks. TabICLv2 ensembles predictions using random column/class shuffles and different preprocessors, as in TabICL. TabArena contains 51 datasets (38 classification with _≤_ 10 classes, 13 regression), evaluated via repeated cross-validation with ROC AUC for binary, log-loss for multiclass, and RMSE for regression. We use 8 estimators for TabICLv2 to match RealTabPFN2.5 in TabArena. TALENT contains 300 datasets (120 binary, 80 multiclass, 100 regression) with 64%/16%/20% train/validation/test splits. Hyperparameters are selected on the validation set using accuracy for classification and RMSE for regression. We use 32 estimators for both TabICLv2 and TabPFN-2.5/RealTabPFN-2.5 in TALENT. 

We primarily report improvability, which measures the average relative error gap to the best method on each dataset, allowing the best method to vary across datasets. Improvability reflects the magnitude of performance differences compared to rank-based metrics. We provide other metrics in Appendices J and K. 

In addition, following TabArena and TALENT, we use the TabICLv1.1 checkpoint for TabICL (Qu et al., 2025), which is TabICL post-trained on an earlier version of our prior. 

**TabICLv2 is state-of-the-art on both benchmarks.** As shown in Figures 1 and 5, TabICLv2 dominates the Pareto fronts of improvability versus runtime. Without any tuning, TabICLv2 surpasses RealTabPFN-2.5 (tuned + ensembled), the current state-of-the-art that is not fully open-source. TabICLv2 also substantially outperforms heavily tuned traditional methods, such as CatBoost and XGBoost, despite requiring orders of magnitude less training time. 



<!-- Start of picture text -->
100 features TabICLv2 vs. TabPFN-2.5<br>11.8x 10.6x NVIDIA GPU<br>10 2 10 (H100 NVL 94GB)<br>10 1 5 AMD(EPYC,CPU48 cores)<br>TabICLv2<br>10 0 TabPFN-2.5 Apple CPU<br>0 10k 30k 50k 0 10k 30k 50k (M3 Pro, 12 cores)<br>Training samples Training samples<br>Figure 6. Runtime comparison between TabICLv2 and<br>TabPFN-2.5 with respect to the number of training samples<br>and hardware. Both use 8 estimators. We use classification with<br>500 test samples.<br>1 . 00<br>80<br>0 . 95<br>60<br>0 . 90<br>40<br>0 . 85<br>20<br>0 . 80<br>TabICLv2-ECOCTabPFN-2.5-ECOCTabICLv2 TabICLRealMLPModernNCATabRResNetMLP-PLRXGBoost<br>(s)<br>time<br>infer<br>+ Speedup<br>Train<br>Accuracy median)<br>time<br>Norm. samples,<br>Inference<br>(s/1K<br>Min-Max<br><!-- End of picture text -->

_Figure 6._ **Runtime comparison between TabICLv2 and TabPFN-2.5 with respect to the number of training samples and hardware.** Both use 8 estimators. We use classification with 500 test samples. 

_Figure 7._ **Normalized accuracy across 12 datasets with more than 10 classes on the TALENT benchmark.** 

**TabICLv2 is consistently faster than TabPFN-2.5.** As shown in Figure 6, for 100 features, TabICLv2 is faster than TabPFN-2.5 across all hardware, with speedups increasing at larger scales: 10.6 _×_ on an H100 GPU at 50K samples. The efficiency gap is even more pronounced on CPU, reaching 11.8 _×_ at just 10K samples. 

**TabICLv2 excels on many-class classification.** TabICLv2 with both the error-correcting output codes (ECOC) wrapper from TabPFNv2 and our native mixed-radix ensembling substantially outperforms all baselines on TALENT datasets with _>_ 10 classes (Figure 7). The ECOC wrapper is slightly better but 3 _×_ slower than our native handling. 

**TabICLv2 scales to large datasets.** As shown in Figure 8, TabICLv2 maintains top rankings across all dataset sizes from 10<sup>3</sup> to 10<sup>5</sup> , outperforming RealTabPFN-2.5 on larger datasets ( _>_ 20K). On even larger datasets (600K) from the TALENT extension, TabICLv2 still performs strongly (Figure 9). These results show that TabICLv2 further extends the frontier of TFMs for natively handling large-scale data. 

### **7. Ablation study** 

We conduct an ablation study to assess the impact of architectural, prior, and pretraining choices (Figure 10). See 

7 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
30 TabICLv2<br>TabPFNv2<br>25 RealTabPFN-2.5<br>RealMLP<br>20 CatBoost<br>15<br>10<br>5<br>0<br>10 3 10 4 10 5<br>Number of samples<br>Rank<br>←<br><!-- End of picture text -->

_Figure 8._ **Model rankings as a function of sample size on TALENT.** The lines show the bootstrap median and 10% / 90% bootstrap confidence intervals of a piecewise linear fit (Qu et al., 2025). 



<!-- Start of picture text -->
Covertype Data Science Good Kiva<br>(581K samples) (671K samples)<br>TabICLv2 XGBoost<br>RealMLP TabICLv2<br>ModernNCA CatBoost<br>MLP-PLR RealMLP<br>XGBoost MLP-PLR<br>ResNet ModernNCA<br>MLP MLP<br>KNN ResNet<br>CatBoost KNN<br>0.90 0.92 0.94 0.96 0.98 0.94 0.95 0.96 0.97<br>Accuracy Accuracy<br><!-- End of picture text -->

_Figure 9._ **Accuracy on two huge classification datasets from the TALENT extension.** TabICLv2 still performs strongly. TabPFN2.5 resulted in out-of-memory errors. 

Appendix C for more ablation results. The performance gap between the reference TabICLv2 checkpoint (dotted black) and its ablation (solid black) is explained by pretraining length: the ablation is trained for 280K steps, whereas the official checkpoint is pretrained for more steps (500K steps for stage 1 plus stages 2 & 3) and uses 8 instead of 4 attention heads in TFcol and TFrow. Interestingly, TabICLv2 matches RealTabPFN-2.5 in log-loss after _≈_ 200K steps, and in _<_ 100K steps in terms of normalized accuracy. 

First, we observe a strong interaction between architecture and prior. Pretraining TabICLv2 with the TabICL prior fails (gray line): the performance remains below TabICL and the validation loss degrades in the second half of the pretraining. This suggests that the TabICLv2 architecture requires higher prior diversity to generalize, perhaps similar to how Ma et al. (2025a) observed that scaling laws can break down with weak synthetic data generators. Additionally, pretraining the TabICL architecture with the TabICLv2 prior (orange) only matches TabICL, indicating that the TabICL architecture has limited ability to exploit increased prior diversity. 

Across metrics (Figure C.1) including normalized accuracy, Elo, and log-loss, the ordering of ablations is consistent. The prior yields the largest effect. Three components provide comparable, significant gains ( _≈_ 100 Elo, 64% win 



<!-- Start of picture text -->
TabICL RealTabPFN-2.5<br>0 . 4<br>TabPFNv2 TabICLv2<br>0 . 3<br>0 . 2<br>0 . 1<br>0 50k 100k 150k 200k 250k<br>Pretraining Step<br>Ref with TabICL prior Ref  − prior filtering<br>Ref with TabICL architecture Ref  − feature grouping<br>Ref  − early target Ref (2 runs)<br>Ref : Muon  → AdamW Ref + QASSMax<br>Log-Loss<br>Min-Max Norm.<br><!-- End of picture text -->

_Figure 10._ **Ablating different components of TabICLv2.** Nonsolid horizontal lines denote performance of official checkpoints; solid lines denote ablated models pretrained for 280K steps. Each ablation modifies one component of the reference model (blue) by adding (+), removing ( _−_ ), or replacing ( _→_ ) it. The reference model corresponds to TabICLv2 without QASSMax and with 4 instead of 8 heads for TFcol and TFicl. Performance metrics are computed on the 60 validation datasets used for TabPFNv2 development (Hollmann et al., 2025, supplementary Table 5). For each dataset, we use up to 2,048 training samples (fewer when the dataset is smaller) and two train/test splits. The AdamW ablation uses regular weight decay for AdamW and a learning rate of 1e-4 following TabICL. Scatterplots display per-step validation performance and reveal decreasing noise as the learning rate decays. 

rate): early target inclusion, Muon instead of AdamW, and QASSMax. Repeated feature grouping and prior filtering yield smaller gains. 

### **8. Limitations** 

TabICLv2 shares common limitations with related models: it does not natively leverage semantic information from column names or textual features, shown to be valuable (Spinaci et al., 2025), but its scalability to large numbers of features suggests it should remain reasonably fast when combined with text embedding models. Additionally, despite improved scalability, datasets with millions of samples remain challenging. Many extensions, such as multi-output regression or handling distribution shifts (Helli et al., 2024) are also left to future work. Due to the lack of established benchmarks, the distributional regression capabilities of TabICLv2 are not evaluated beyond toy datasets (Appendix I.9). Adding missing indicators (Le Morvan & Varoquaux, 2025) or introducing missingness during pretraining may improve the handling of missing values, which are currently imputed by the mean, but remain unexplored. Finally, hyperparameter tuning or fine-tuning (Rubachev 

8 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

et al., 2025) could further improve the performance at the cost of increased runtime, but is not explored here. 

### **9. Conclusion** 

TabICLv2 represents a large step forward in tabular foundation models (TFMs), as it achieves state-of-the-art performance and redefines the native scalability of TFMs. We commit to fully open-sourcing everything to democratize access to state-of-the-art TFMs. With its moderate pretraining and inference cost, TabICLv2 provides an excellent basis for future adaptations. In addition, we prioritize out-of-the-box performance and principled innovations over fine-tuning on real data (Garg et al., 2025) or scaling up with deeper (Grinsztajn et al., 2025) or wider (Ma et al., 2025a; Zhang et al., 2025a) architectures. We hope that TabICLv2 motivates continued innovation towards smaller, faster, better models. 

### **Acknowledgements** 

We thank Fabian Schaipp for encouraging us to try his Muon implementation. We thank Tizian Wenzel and Ingo Steinwart for helpful discussions on theory. We are grateful to the authors of the TALENT benchmark for providing information about the TALENT extension. We also thank the LimiX team, especially Xingxuan Zhang and Peng Cui, for their generous support in providing computational resources. 

This work was performed using HPC resources from GENCI–IDRIS (Grant 2024-AD011014864R1, 2024AD011016033, 2025-AD011016033R1). 

### **Contribution Statement** 

JQ and DH contributed equally to this work (co-first authors). JQ developed the regression framework and multiclass extensions, enabled scaling to large _n_ through optimized implementation and the design of QASSMax, conducted the majority of experiments and benchmark evaluations, and prepared production-ready code. DH conceived and implemented the new generative prior and other core architectural and pretraining enhancements, conducted smallscale experiments, and implemented nanotabicl. MLM and JQ managed larger-scale pretraining runs and the systematic evaluation of model variants. All authors participated in the weekly technical steering, experimental design, and iterative refinement of the manuscript. 

### **Impact Statement** 

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 

### **References** 

- Balazadeh, V., Kamkari, H., Thomas, V., Ma, J., Li, B., Cresswell, J. C., and Krishnan, R. CausalPFN: Amortized causal effect estimation via in-context learning. In _The Thirty-ninth Annual Conference on Neural Information Processing Systems_ , 2025. 

- Barlow, R. E. and Brunk, H. D. The isotonic regression problem and its dual. _Journal of the American Statistical Association_ , 67(337):140–147, 1972. 

- Beaglehole, D., Holzmuller,¨ D., Radhakrishnan, A., and Belkin, M. xRFM: Accurate, scalable, and interpretable feature learning models for tabular data. In _AITD 2025– Workshop on AI for Tabular Data_ , 2025. 

- Beirlant, J., Goegebeur, Y., Segers, J., and Teugels, J. L. _Statistics of extremes: theory and applications_ . John Wiley & Sons, 2006. 

- Bell, J. Trace class operators and Hilbert-Schmidt operators. _Lecture Notes_ , 2016. 

- Best, M. J. and Chakravarti, N. Active set algorithms for isotonic regression; a unifying framework. _Mathematical Programming_ , 47(1):425–439, 1990. 

- Bondell, H. D., Reich, B. J., and Wang, H. Noncrossing quantile regression curve estimation. _Biometrika_ , 97(4): 825–838, 2010. 

- Bonet, D., Montserrat, D. M., Giro-i Nieto, X., and Ioan-´ nidis, A. G. Hyperfast: Instant classification for tabular data. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pp. 11114–11123, 2024. 

- Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., and Kasneci, G. Deep neural networks and tabular data: A survey. _IEEE transactions on neural networks and learning systems_ , 35(6):7499–7519, 2022. 

- Bouadi, M., Seth, P., Tanna, A., and Sankarapu, V. K. Orionmsp: Multi-scale sparse attention for tabular in-context learning. _arXiv preprint arXiv:2511.02818_ , 2025. 

- Breejen, F. d., Bae, S., Cha, S., and Yun, S.-Y. Fine-tuned in-context learning transformers are excellent tabular data classifiers. _arXiv preprint arXiv:2405.13396_ , 2024. 

- Busing, F. M. Monotone regression: A simple and fast O(n) PAVA implementation. _Journal of Statistical Software_ , 102:1–25, 2022. 

- Chen, L., Li, J., Liang, K., Su, B., Xie, C., Pierse, N. W., Liang, C., Lao, N., and Liu, Q. Cautious weight decay. _arXiv preprint arXiv:2510.12402_ , 2025a. 

9 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

- Chen, S., Lin, Z., Polyanskiy, Y., and Rigollet, P. Critical attention scaling in long-context transformers. _ArXiv_ , abs/2510.05554, 2025b. 

- Chernozhukov, V., Fernandez-Val,´ I., and Galichon, A. Quantile and probability curves without crossing. _Econometrica_ , 78(3):1093–1125, 2010. 

- Da Costa, N., Pfortner,¨ M., Da Costa, L., and Hennig, P. Sample path regularity of gaussian processes from the covariance kernel. _arXiv preprint arXiv:2312.14886_ , 2023. 

- Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. _ArXiv_ , abs/2307.08691, 2023. 

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and R’e, C. Flashattention: Fast and memory-efficient exact attention with io-awareness. _ArXiv_ , abs/2205.14135, 2022. 

- De Vito, E., Mucke,¨ N., and Rosasco, L. Reproducing kernel Hilbert spaces on manifolds: Sobolev and diffusion spaces. _Analysis and Applications_ , 19(03):363–396, 2021. 

- Defazio, A., Yang, X., Mehta, H., Mishchenko, K., Khaled, A., and Cutkosky, A. The road less scheduled. _Neural Information Processing Systems_ , 2024. 

- Dietterich, T. G. and Bakiri, G. Solving multiclass learning problems via error-correcting output codes. _Journal of artificial intelligence research_ , 2:263–286, 1994. 

- Dong, H., Zhang, P., Lu, M., Shen, Y., and Ke, G. Machinelearninglm: Scaling many-shot in-context learning via continued pretraining. _arXiv preprint arXiv:2509.06806_ , 2025. 

- Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., and Smola, A. Autogluon-tabular: Robust and accurate automl for structured data. _arXiv preprint arXiv:2003.06505_ , 2020. 

- Erickson, N., Purucker, L., Tschalzev, A., Holzmuller, D.,¨ Desai, P. M., Salinas, D., and Hutter, F. TabArena: A living benchmark for machine learning on tabular data. In _Neural Information Processing Systems_ , 2025. 

- Feuer, B., Schirrmeister, R. T., Cherepanova, V., Hegde, C., Hutter, F., Goldblum, M., Cohen, N., and White, C. Tunetables: Context optimization for scalable priordata fitted networks. _Advances in Neural Information Processing Systems_ , 37:83430–83464, 2024. 

- Gardner, J., Perdomo, J. C., and Schmidt, L. Large scale transfer learning for tabular data via language modeling. _Advances in Neural Information Processing Systems_ , 37: 45155–45205, 2024. 

- Garg, A., Ali, M., Hollmann, N., Purucker, L., Muller, S.,¨ and Hutter, F. Real-tabPFN: Improving tabular foundation models via continued pre-training with real-world data. In _1st ICML Workshop on Foundation Models for Structured Data_ , 2025. 

- Geurts, P., Ernst, D., and Wehenkel, L. Extremely randomized trees. _Machine learning_ , 63(1):3–42, 2006. 

- Grinsztajn, L., Oyallon, E., and Varoquaux, G. Why do treebased models still outperform deep learning on typical tabular data? _Neural Information Processing Systems_ , 35: 507–520, 2022. 

- Grinsztajn, L., Floge, K., Key, O., Birkel, F., Jund, P., Roof,¨ B., Jager,¨ B., Safaric, D., Alessi, S., Hayler, A., et al. Tabpfn-2.5: Advancing the state of the art in tabular foundation models. _arXiv preprint arXiv:2511.08667_ , 2025. 

- Hassan, C., Loka, N., Li, C.-Y., Huang, D., Chang, P. E., Yang, Y., Silvestrin, F., Kaski, S., and Acerbi, L. Efficient autoregressive inference for transformer probabilistic models. _arXiv preprint arXiv:2510.09477_ , 2025. 

- Hegselmann, S., Buendia, A., Lang, H., Agrawal, M., Jiang, X., and Sontag, D. Tabllm: Few-shot classification of tabular data with large language models. In _International conference on artificial intelligence and statistics_ , pp. 5549–5581. PMLR, 2023. 

- Helli, K., Schnurr, D., Hollmann, N., Muller, S., and Hutter,¨ F. Drift-resilient tabpfn: In-context learning temporal distribution shifts on tabular data. _Advances in Neural Information Processing Systems_ , 37:98742–98781, 2024. 

- Hollmann, N., Muller,¨ S., Eggensperger, K., and Hutter, F. Tabpfn: A transformer that solves small tabular classification problems in a second. _arXiv preprint arXiv:2207.01848_ , 2022. 

- Hollmann, N., Muller, S., Purucker, L., Krishnakumar, A.,¨ Korfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter,¨ F. Accurate predictions on small data with a tabular foundation model. _Nature_ , 637(8045):319–326, 2025. 

- Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R., Gama, J. a., and Bizarro, P. Turning the tables: Biased, imbalanced, dynamic tabular datasets for ml evaluation. In _Neural Information Processing Systems_ , 2022. 

- Jordan, K., Bernstein, J., Rappazzo, B., @fernbear.bsky.social, Vlado, B., Jiacheng, Y., Cesista, F., Koszarsky, B., and @Grad62304977. modded-nanogpt: Speedrunning the nanogpt baseline, 2024a. 

- Jordan, K., Jin, Y., Boza, V., Jiacheng, Y., Cesista, F., Newhouse, L., and Bernstein, J. Muon: An optimizer for hidden layers in neural networks, 2024b. 

10 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

- Kolberg, C., Eggensperger, K., and Pfeifer, N. Tabpfn-wide: Continued pre-training for extreme feature counts. _arXiv preprint arXiv:2510.06162_ , 2025. 

- Kumaraswamy, P. A generalized probability density function for double-bounded random processes. _Journal of hydrology_ , 46(1-2):79–88, 1980. 

- Lam, S. K., Pitrou, A., and Seibert, S. Numba: a llvm-based python jit compiler. In _LLVM ’15_ , 2015. 

- Le Morvan, M. and Varoquaux, G. Imputation for prediction: beware of diminishing returns. In _The Thirteenth International Conference on Learning Representations_ , 2025. 

- Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., and Teh, Y. W. Set transformer: A framework for attention-based permutation-invariant neural networks. In _International conference on machine learning_ , pp. 3744–3753. PMLR, 2019. 

- Liu, J., Su, J., Yao, X., Jiang, Z., Lai, G., Du, Y., Qin, Y., Xu, W., Lu, E., Yan, J., et al. Muon is scalable for llm training. _arXiv preprint arXiv:2502.16982_ , 2025. 

- Liu, S.-Y. and Ye, H.-J. Tabpfn unleashed: A scalable and effective solution to tabular classification problems. _arXiv preprint arXiv:2502.02527_ , 2025. 

- Lukic, M. and Beder, J.´ Stochastic processes with sample paths in reproducing kernel Hilbert spaces. _Transactions of the American Mathematical Society_ , 353(10):3945– 3969, 2001. 

- Ma, J., Thomas, V., Hosseinzadeh, R., Labach, A., Cresswell, J. C., Golestan, K., Yu, G., Caterini, A. L., and Volkovs, M. TabDPT: Scaling tabular foundation models on real data. In _Neural Information Processing Systems_ , 2025a. 

- Ma, Y., Frauen, D., Javurek, E., and Feuerriegel, S. Foundation models for causal inference via prior-data fitted networks. _arXiv preprint arXiv:2506.10914_ , 2025b. 

- Mueller, A. C., Curino, C. A., and Ramakrishnan, R. Mothernet: Fast training and inference via hyper-network transformers. In _The Thirteenth International Conference on Learning Representations_ , 2024. 

- Muller, S., Hollmann, N., Arango, S. P., Grabocka, J., and¨ Hutter, F. Transformers can do bayesian inference. _arXiv preprint arXiv:2112.10510_ , 2021. 

- Nakanishi, K. M. Scalable-softmax is superior for attention. _arXiv preprint arXiv:2501.19399_ , 2025. 

- Pagliardini, M., Ablin, P., and Grangier, D. The AdEMAMix optimizer: Better, faster, older. In _International Conference on Learning Representations_ , 2025. 

- Park, Y., Maddix, D., Aubet, F.-X., Kan, K., Gasthaus, J., and Wang, Y. Learning quantile functions without quantile crossing for distribution-free time series forecasting. In _International Conference on Artificial Intelligence and Statistics_ , 2022. 

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al. Scikit-learn: Machine learning in python. _the Journal of machine Learning research_ , 12:2825–2830, 2011. 

- Peng, B., Quesnelle, J., Fan, H., and Shippole, E. Yarn: Efficient context window extension of large language models. _ArXiv_ , abs/2309.00071, 2023. 

- Peters, B., Niculae, V., and Martins, A. F. T. Sparse sequence-to-sequence models. _ArXiv_ , abs/1905.05702, 2019. 

- Pfefferle, A., Hog, J., Purucker, L., and Hutter, F. nanotabpfn: A lightweight and educational reimplementation of tabpfn. In _EurIPS 2025 Workshop: AI for Tabular Data_ , 2025. 

- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. Catboost: unbiased boosting with categorical features. In _Neural Information Processing Systems_ , 2018. 

- Qiu, Z., Wang, Z., Zheng, B., Huang, Z., Wen, K., Yang, S., Men, R., Yu, L., Huang, F., Huang, S., Liu, D., Zhou, J., and Lin, J. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free. In _Neural Information Processing Systems_ , 2025. 

- Qu, J., Holzmuller,¨ D., Varoquaux, G., and Le Morvan, M. TabICL: A tabular foundation model for in-context learning on large data. In _International Conference on Machine Learning_ , 2025. 

- Rahimi, A. and Recht, B. Random features for large-scale kernel machines. In _Neural Information Processing Systems_ , 2007. 

- Ramapuram, J., Danieli, F., Dhekane, E. G., Weers, F., Busbridge, D., Ablin, P., Likhomanenko, T., Digani, J., Gu, Z., Shidani, A., and Webb, R. Theory, analysis, and best practices for sigmoid self-attention. _ArXiv_ , abs/2409.04431, 2024. 

- Robertson, J., Reuter, A., Guo, S., Hollmann, N., Hutter, F., and Scholkopf, B.¨ Do-PFN: In-context learning for causal effect estimation. In _Neural Information Processing Systems_ , 2025. 

11 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

- Rubachev, I., Kotelnikov, A., Kartashev, N., and Babenko, A. On finetuning tabular foundation models. _arXiv preprint arXiv:2506.08982_ , 2025. 

- Schaipp, F. Optimization benchmark for diffusion models on dynamical systems. In _EurIPS 2025 Workshop on Principles of Generative Modeling (PriGM)_ , 2025. 

- Sergazinov, R. and Yin, S.-A. Chunked tabpfn: Exact training-free in-context learning for long-context tabular data. _arXiv preprint arXiv:2509.00326_ , 2025. 

- Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. _ArXiv_ , abs/2407.08608, 2024. 

- Spinaci, M., Polewczyk, M., Schambach, M., and Thelin, S. ConTextTab: A semantics-aware tabular in-context learner. In _Neural Information Processing Systems_ , 2025. 

- Steinwart, I. _Support Vector Machines_ . Springer, 2008. 

- Steinwart, I. When does a gaussian process have its paths in a reproducing kernel hilbert space? _arXiv preprint arXiv:2407.11898_ , 2024. 

- Su, J., Lu, Y., Pan, S., Wen, B., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. _ArXiv_ , abs/2104.09864, 2021. 

   - Wilson, A. G. and Izmailov, P. Bayesian deep learning and a probabilistic perspective of generalization. _Advances in neural information processing systems_ , 33:4697–4708, 2020. 

   - Xu, D., Cirit, O., Asadi, R., Sun, Y., and Wang, W. Mixture of in-context prompters for tabular pfns. _arXiv preprint arXiv:2405.16156_ , 2024. 

   - Ye, H.-J., Liu, S.-Y., Cai, H.-R., Zhou, Q.-L., and Zhan, D.-C. A closer look at deep learning methods on tabular datasets. _arXiv preprint arXiv:2407.00956_ , 2024. 

   - Zhang, X., Chang, X., Li, M., Roy-Chowdhury, A. K., Chen, J., and Oymak, S. Selective attention: Enhancing transformer through principled context control. _ArXiv_ , abs/2411.12892, 2024. 

   - Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu, L., Faloutsos, C., Mahoney, M. W., et al. Mitra: Mixed synthetic priors for enhancing tabular foundation models. _arXiv preprint arXiv:2510.21204_ , 2025a. 

   - Zhang, X., Ren, G., Yu, H., Yuan, H., Wang, H., Li, J., Wu, J., Mo, L., Mao, L., Hao, M., et al. Limix: Unleashing structured-data modeling capability for generalist intelligence. _arXiv preprint arXiv:2509.03505_ , 2025b. 

- Thomas, V., Ma, J., Hosseinzadeh, R., Golestan, K., Yu, G., Volkovs, M., and Caterini, A. Retrieval & fine-tuning for in-context tabular models. _Neural Information Processing Systems_ , 2024. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In _Neural Information Processing Systems_ , 2017. 

- Vasylenko, P., Pitorro, H., Martins, A. F., and Treviso, M. Long-context generalization with sparse attention. _arXiv preprint arXiv:2506.16640_ , 2025. 

- Velickoviˇ c, P., Perivolaropoulos, C., Barbero, F., and Pas-´ canu, R. Softmax is not enough (for sharp size generalisation). _arXiv preprint arXiv:2410.01104_ , 2024. 

- Vetter, J., Gloeckler, M., Gedon, D., and Macke, J. H. Effortless, simulation-efficient bayesian inference using tabular foundation models. In _Neural Information Processing Systems_ , 2025. 

- Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., et al. Scipy 1.0: fundamental algorithms for scientific computing in python. _Nature methods_ , 17(3):261–272, 2020. 

12 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

# **Appendices** 

### **Contents** 

|**A More architecture details about TabICLv2**|**15**|
|---|---|
|A.1<br>Repeated feature grouping<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>15|
|A.2<br>Compression then ICL<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>15|
|A.3<br>Many-class classifcation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>15|
|A.4<br>Model confguration. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>17|
|**B**<br>**More pretraining details about TabICLv2**|**18**|
|B.1<br>Three pretraining stages . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>18|
|B.2<br>Speed and memory optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>18|
|**C Additional ablation results**|**20**|
|**D Other things we tried**|**20**|
|**E**<br>**Details on the prior**|**22**|
|E.1<br>Differences to previous priors. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>22|
|E.2<br>Sampling correlated scalars . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
|E.3<br>Random dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
|E.4<br>Random graph. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
|E.5<br>Random node function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
|E.6<br>Random converter . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>24|
|E.7<br>Random multi-function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>24|
|E.8<br>Random functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>24|
|E.9<br>Random activations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>26|
|E.10 Random matrix . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>27|
|E.11 Random weights<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>27|
|E.12 Random points<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>27|
|E.13 Postprocessing<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>27|
|E.14 Filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>28|
|**F**<br>**Plots for the prior**|**28**|
|**G Path smoothness for Gaussian processes**|**34**|
|**H Inference optimization for TabICLv2**|**36**|
|H.1<br>Effcient attention computation via selective query-key-value projections . . . . . . . . .|. . . . . . . . .<br>36|



13 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

|H.2|Offoading technique . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>36|
|---|---|---|
|**I**<br>**Qua**|**ntile distribution**|**39**|
|I.1|Problem setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>39|
|I.2|Quantile function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>39|
|I.3|Quantile crossing correction<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>40|
|I.4|Tail parameter estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>41|
|I.5|Cumulative distribution function (CDF) . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>41|
|I.6|Probability density function (PDF) . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>42|
|I.7|Continuous ranked probability score (CRPS). . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>42|
|I.8|Moment calculations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>44|
|I.9|Empirical validation on synthetic regression tasks . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>45|
|**J**<br>**Deta**|**iled results on the TabArena benchmark**|**48**|
|J.1|Aggregation metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>48|
|J.2|Results on all datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>48|
|J.3|Results on binary classifcation datasets<br>. . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>52|
|J.4|Results on multiclass classifcation datasets<br>. . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>53|
|J.5|Results on regression datasets<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>55|
|**K Deta**|**iled results on the TALENT benchmark**|**56**|
|K.1|Benchmark overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>56|
|K.2|Results on all datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>56|
|K.3|Results on binary classifcation datasets<br>. . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>59|
|K.4|Results on multiclass classifcation datasets (_≤_10 classes)<br>. . . . . . . . . . . . .|. . . . . . . . . . . .<br>62|
|K.5|Results on multiclass classifcation datasets (_>_10 classes)<br>. . . . . . . . . . . . .|. . . . . . . . . . . .<br>65|
|K.6|Results on regression datasets<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>67|
|K.7|Results on small datasets with less than 10K samples . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>70|
|K.8|Results on large datasets with more than 10K samples . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>71|
|K.9|Model rankings with respect to meta-features<br>. . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>73|



14 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

### **A. More architecture details about TabICLv2** 

#### **A.1. Repeated feature grouping** 

The generalization of our repeated feature grouping pattern from groups of three columns to groups of _k_ columns among _m_ columns is to define group _i_ (for _i ≥_ 0) as 



Theorem A.1 shows that whenever _m ≥_ 2<sup>_k_</sup> , no pair of columns occurs together in two groups. In principle, larger _k_ yields a more expressive architecture, as the linear layer can always learn to ignore some of the columns. In practice, we use _k_ = 3 since larger values did not seem beneficial in initial experiments. 

**Lemma A.1** (Intersections of feature groups) **.** _For k ≥_ 0 _, m ≥_ 1 _, i ∈{_ 0 _, . . . , m −_ 1 _}, define the set_ 



_Then, if m ≥_ 2<sup>_k_</sup> _, we have |Ii,k,m ∩ Ij,k,m| ≤_ 1 _for all_ 0 _≤ i < j ≤ m −_ 1 _._ 

_Proof._ Suppose that _|Ii,k,m ∩ Ij,k,m| ≥_ 2. But then, by shift invariance, for all _n ∈_ Z, _|Ii_ + _n_ mod _m,k,m ∩ Ij_ + _n_ mod _m,k,m| ≥_ 2. Hence, by a suitable shift, we can assume without loss of generality that _i_ = 0 and _j ≤ m/_ 2 _≤ m −_ 2<sup>_k−_1</sup> . Hence, the modulo in the definitions of _Ii,k,m_ and _Ij,k,m_ does nothing and we can find _a, b, c, d ∈{_ 0 _, . . . , k −_ 1 _}_ with _a_ = _b_ and 



which in particular implies 



yielding 2<sup>_b_</sup> + 2<sup>_d_</sup> = 2<sup>_a_</sup> + 2<sup>_c_</sup> . But this means that _{b, d}_ = _{a, c}_ . From _i < j_ we know that _a > c_ and _b > d_ , implying _a_ = _b_ and _c_ = _d_ , contradicting our previous assumption. 

#### **A.2. Compression then ICL** 

**Column-wise embedding.** Column-wise embedding processes each column through a set transformer TFcol (Lee et al., 2019). Its core is _induced self-attention_ with _k_ inducing vectors that proceeds in two stages: the first stage aggregates input information into inducing vectors and the second broadcasts back to the input, reducing complexity from _O_ ( _n_<sup>2</sup> ) to _O_ ( _nk_ ). We make three improvements: (a) we directly use the outputs of TFcol as feature embeddings, while TabICL applies an additional transformation; (b) we apply our query-aware scalable softmax (QASSMax) to the first stage of induced self-attention; and (c) TFcol is essentially an in-context learner operating within each column thanks to target-aware embedding. 

**Row-wise interaction.** Following TabICL (Qu et al., 2025), we prepend four learnable [CLS] tokens to each row and process them through a transformer TFrow with RoPE (Su et al., 2021). The outputs of the [CLS] tokens are concatenated to form a 4 _d_ -dimensional row embeddings, effectively collapsing the feature dimension. 

**Dataset-wise in-context learning.** Training row embeddings are combined with target embeddings. A transformer TFicl processes all embeddings, where test samples attend only to training samples. A two-layer MLP converts the outputs into target predictions. We apply QASSMax to TFicl to improve its long-context generalization. 

#### **A.3. Many-class classification** 

Like many tabular foundation models, TabICLv2 is pretrained on classification tasks with up to 10 classes. For tasks with more classes, TabICL (Qu et al., 2025) proposed hierarchical classification during the ICL stage, which recursively partitions classes into subproblems with at most 10 classes each. However, our target-aware embedding introduces label information before hierarchical partitioning occurs. Since our label encoder supports at most 10 classes, we need a mechanism to handle many-class scenarios at this early stage. We propose _mixed-radix ensembling_ , which generates multiple simplified views of the original labels, each containing at most 10 classes. The key idea is to decompose the class label using a mixed-radix number system, where each digit corresponds to a different view of the classification problem. 

15 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

**Computing balanced bases.** For a task with _C >_ 10 classes, we first compute a sequence of balanced bases [ _k_ 0 _, k_ 1 _, . . . , kD−_ 1] satisfying two constraints: 

1. Each base is bounded: _ki ≤_ 10 for all _i_ 

2. The product covers all classes:<sup>�</sup><sup>_D_</sup> _i_ =0<sup>_−_1</sup><sup>_ki≥C_</sup> 

We select bases to be as balanced as possible (i.e., _ki ≈ kj_ ) to ensure each view captures roughly equal discriminative information. The number of views _D_ is minimized subject to these constraints. 

**Mixed-radix label encoding.** Given the bases [ _k_ 0 _, . . . , kD−_ 1], each class label _y ∈{_ 0 _,_ 1 _, . . . , C −_ 1 _}_ is re-encoded into _D_ views using mixed-radix decomposition: 



This is analogous to representing a number in a mixed-radix positional system, where each “digit” _y_<sup>(</sup><sup>_i_)</sup> takes values in _{_ 0 _,_ 1 _, . . . , ki −_ 1 _}_ . Consider a 16-class problem ( _C_ = 16) with bases [ _k_ 0 _, k_ 1] = [4 _,_ 4]: 

**View 0** : _y_<sup>(0)</sup> = _⌊y/_ 4 _⌋_ partitions classes into consecutive blocks: 

- Classes _{_ 0 _,_ 1 _,_ 2 _,_ 3 _} →_ 0 

- Classes _{_ 4 _,_ 5 _,_ 6 _,_ 7 _} →_ 1 

- Classes _{_ 8 _,_ 9 _,_ 10 _,_ 11 _} →_ 2 

- Classes _{_ 12 _,_ 13 _,_ 14 _,_ 15 _} →_ 3 

**View 1** : _y_<sup>(1)</sup> = _y_ mod 4 partitions classes by remainder: 

- Classes _{_ 0 _,_ 4 _,_ 8 _,_ 12 _} →_ 0 

- Classes _{_ 1 _,_ 5 _,_ 9 _,_ 13 _} →_ 1 

- Classes _{_ 2 _,_ 6 _,_ 10 _,_ 14 _} →_ 2 

- Classes _{_ 3 _,_ 7 _,_ 11 _,_ 15 _} →_ 3 

Each view creates a different grouping of the original classes, and no single view can distinguish all 16 classes. However, the combination of both views uniquely identifies each class: the pair ( _y_<sup>(0)</sup> _, y_<sup>(1)</sup> ) forms a bijection with the original label _y_ . 

**Ensemble aggregation.** For each view _i_ , we run the column-wise transformer TFcol with the corresponding re-encoded labels: 



where _E_ 1 denotes the embeddings before label injection and EmbedTAE is the target-aware embedding layer. The final output is the average across all views: 



**Relationship to error-correcting output codes.** Our approach is inspired by error-correcting output codes (ECOC) (Dietterich & Bakiri, 1994), which decomposes multi-class classification into multiple binary problems. However, unlike ECOC which uses binary codes and trains separate classifiers, our method: (1) uses _k_ -ary codes with _k ≤_ 10 to match our pretrained label encoder capacity, (2) operates at the embedding level rather than the prediction level and averages embeddings rather than combining binary predictions. 

**Combined with hierarchical classification.** Hierarchical classification operates at the ICL stage, while mixed-radix ensembling handles many-class scenarios at the column-wise embedding stage. Together, they enable TabICLv2 to handle classification tasks with an arbitrary number of classes without retraining. 

16 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **A.4. Model configuration** 

TabICLv2 adopts an architecture similar to TabICL, consisting of column-wise embedding through a Set Transformer TFcol, row-wise interaction through a Transformer encoder TFrow, and dataset-wise in-context learning through a Transformer encoder TFicl. We train separate checkpoints for classification and regression tasks. Table A.1 summarizes the key architectural differences between the two models. 

**Classification model.** TFcol consists of three induced self-attention blocks with 128 inducing vectors, model dimension _d_ = 128, and 8 attention heads. The target-aware embedding EmbedTAE is a learnable lookup table providing class embeddings for 10 classes. 

TFrow is a 3-layer Transformer encoder with model dimension _d_ = 128 and 8 attention heads. It uses 4 learnable [CLS] tokens to aggregate feature-wise information into a single row representation. 

TFicl is a 12-layer Transformer encoder with model dimension _d_ = 512 and 8 attention heads. The ICL-stage target embedding EmbedICL is also a learnable lookup table for 10 classes. 

All Transformer attention blocks use pre-norm layer normalization with learnable weights and biases and GELU activations. The feedforward modules use a dimension expansion factor of 2. 

The final prediction head is a 2-layer MLP with hidden dimension 1024 and output dimension 10. 

For QASSMax, MLPbase : R _→_ R<sup>_H×d_head</sup> and MLPgate : R<sup>_d_head</sup> _→_ R<sup>_d_head</sup> are both 2-layer MLPs with hidden dimension 64 and GELU activation. The last layer of MLPgate is initialized to zero, ensuring the initial modulation is identity. 

**Regression model.** Adapting TabICLv2 for regression requires minimal architectural changes: the target-aware embedding EmbedTAE and ICL-stage target embedding EmbedICL use linear layers to embed continuous targets instead of class lookup tables, and the final MLP outputs 999 quantile predictions instead of class probabilities. 

Compared to the classification model, the regression model uses bias-free layer normalizations with learnable weights only. 

_Table A.1._ Model configuration for classification and regression. 

|**Component**|**Classifcation**|**Regression**|
|---|---|---|
|_TFcol (Column-wise em_|_bedding)_||
|Layers|3|3|
|Inducing vectors|128|128|
|<br>Model dimension|128|128|
|Attention heads|8|8|
|_TFrow (Row-wise inter_|_action)_||
|Layers|3|3|
|Model dimension|128|128|
|Attention heads|8|8|
|[CLS]tokens|4|4|
|_TFicl (In-context learn_|_ing)_||
|Layers|12|12|
|Model dimension|512|512|
|Attention heads|8|8|
|_Target embedding_|||
|EmbedTAE|Lookup (10 classes)|Linear|
|EmbedICL|Lookup (10 classes)|Linear|
|_Prediction head_|||
|Hidden dimension|1024|1024|
|Output dimension|10|999|
|_Other settings_|||
|LayerNorm bias|Yes|No|
|FFN expansion|2_×_|2_×_|
|Activation|GELU|GELU|



17 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

### **B. More pretraining details about TabICLv2** 

#### **B.1. Three pretraining stages** 

Following TabICL (Qu et al., 2025), we adopt a three-stage pretraining curriculum that progressively increases the sample size of synthetic datasets with batch size of 64 as follows: 

- **Stage 1** : 500K steps with 1,024 samples per dataset. 

- **Stage 2** : 40K steps with 400–10,240 samples (log-uniform). 

- **Stage 3** : 10K steps with 400–60,000 samples (log-uniform). 

Figure B.1 shows the performance of TabICLv2 after each stage on the TALENT benchmark. Each stage yields consistent improvements: on all datasets (Figure B.1a), average rank improves from 9.94 (Stage 1) to 5.69 (Stage 2) to 5.41 (Stage 3). The gains are most pronounced on large datasets with more than 10K samples (Figure B.1c): Stage 1 achieves only rank 14.91, comparable to XGBoost (14.60), but Stage 2 dramatically improves to 5.50 and Stage 3 further reaches 4.71, substantially outperforming all baselines including RealTabPFN-2.5 (6.35). This demonstrates that exposure to larger synthetic datasets during pretraining is crucial for generalization to real-world large-scale datasets. 

#### **B.2. Speed and memory optimization** 

Automatic mixed precision is always used. For stage 3, we enable gradient checkpointing when the sample size exceeds 20K to avoid the out-of-memory error. For stages 2 and 3, we use FlashAttention-3 (Shah et al., 2024), which provides an average 1.3 _×_ speedup over FlashAttention-2 (Dao, 2023) on large-scale pretraining. 

18 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
30 25 20 15 10 5<br>SwitchTab [28.91] [5.41] TabICLv2-Stage3<br>TabNet [27.51] [5.69] TabICLv2-Stage2<br>GrowNet [26.39] [6.16] RealTabPFN-2.5<br>TabTransformer [25.64] [6.53] TabPFN-2.5<br>Linear [24.86] [9.61] LimiX<br>DANets [24.71] [9.94] TabICLv2-Stage1<br>KNN [23.54] [10.17] TabPFNv2<br>NODE [21.14] [12.20] TabICL<br>TANGOS [20.77] [13.11] ModernNCA<br>SNN [20.23] [13.15] CatBoost<br>PTaRL [19.66] [13.43] RealMLP<br>ExcelFormer [19.25] [14.09] TabR<br>MLP [19.20] [14.30] LightGBM<br>ResNet [19.08] [15.27] XGBoost<br>AutoInt [19.05] [16.93] FT-Transformer<br>DCNv2 [18.91] [17.40] MLP-PLR<br>RandomForest [18.76]<br><!-- End of picture text -->



<!-- Start of picture text -->
(a)  Results on all datasets<br>25 20 15 10 5<br>TabNet [28.59] [5.79] TabICLv2-Stage2<br>SwitchTab [28.16] [5.79] TabICLv2-Stage3<br>GrowNet [25.69] [6.06] RealTabPFN-2.5<br>TabTransformer [24.81] [6.35] TabPFN-2.5<br>DANets [24.36] [7.24] TabICLv2-Stage1<br>Linear [23.41] [7.69] LimiX<br>KNN [22.49] [8.60] TabPFNv2<br>NODE [21.08] [11.32] TabICL<br>ExcelFormer [20.48] [13.82] ModernNCA<br>SNN [20.38] [14.01] CatBoost<br>PTaRL [20.30] [14.30] LightGBM<br>AutoInt [19.99] [15.14] RealMLP<br>DCNv2 [19.90] [15.63] XGBoost<br>TANGOS [19.89] [15.80] TabR<br>MLP [19.73] [17.13] RandomForest<br>ResNet [19.44] [18.33] FT-Transformer<br>MLP-PLR [19.32]<br><!-- End of picture text -->



<!-- Start of picture text -->
(b)  Results on small datasets with less than 10K samples<br>30 25 20 15 10 5<br>SwitchTab [30.30] [4.71] TabICLv2-Stage3<br>GrowNet [27.68] [5.50] TabICLv2-Stage2<br>Linear [27.54] [6.35] RealTabPFN-2.5<br>TabTransformer [27.18] [6.86] TabPFN-2.5<br>TabNet [25.50] [10.28] RealMLP<br>KNN [25.47] [10.93] TabR<br>DANets [25.35] [11.57] CatBoost<br>TANGOS [22.42] [11.78] ModernNCA<br>RandomForest [21.78] [13.07] TabPFNv2<br>NODE [21.26] [13.16] LimiX<br>SNN [19.95] [13.84] TabICL<br>PTaRL [18.47] [13.84] MLP-PLR<br>ResNet [18.42] [14.30] LightGBM<br>MLP [18.23] [14.35] FT-Transformer<br>AutoInt [17.32] [14.60] XGBoost<br>DCNv2 [17.09] [14.91] TabICLv2-Stage1<br>ExcelFormer [16.99]<br><!-- End of picture text -->

_(c)_ Results on large datasets with more than 10K samples 

_Figure B.1._ **Critical difference diagram on the TALENT benchmark for TabICLv2 pretrained on three stages.** 

19 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

### **C. Additional ablation results** 

In the main text (Section 7), we present ablation results based on log-loss. Here, we provide additional results using normalized accuracy (Figure C.1a) and Elo (Figure C.1b). Across all three metrics, the ordering of ablations remains consistent, leading to the same conclusions about the contribution of each component. 

We additionally perform two more ablation experiments: 

**Increasing model depth.** We ablate the effect of increasing model depth (light red line in Figure C.1): 4 layers for TFcol and TFrow (instead of 3) and 18 layers for TFicl (instead of 12). Based on log-loss, this deeper model shows no clear improvement over the reference model. However, normalized accuracy and Elo suggest a slight improvement toward the end of pretraining. This marginal gain is likely due to insufficient pretraining for the larger model to fully converge. Nonetheless, since our goal is to achieve state-of-the-art performance through principled innovations rather than simply scaling up model size, we did not pursue this direction further. 

**Adding noise to the prior.** Following TabPFNv2 (Hollmann et al., 2025), which adds Gaussian noise at each edge of the causal graph during synthetic data generation to introduce uncertainty, we experimented with incorporating similar noise into our prior (green solid line in Figure C.1). However, this modification has negligible impact on performance across all metrics. 

### **D. Other things we tried** 

Here, we want to describe some other things that we tried but did not end up using, mostly in smaller-scale experiments and without careful analysis. We hope that it can serve as anecdotal evidence to other model developers. Generally, the results of pretraining runs are somewhat noisy, so these observations have to be taken with at least one grain of salt. Reducing the pretraining noise could itself be a useful contribution of future research. 

**Pretraining.** Contrary to Ma et al. (2025a), we did not see a benefit from using schedule-free AdamW (Defazio et al., 2024) over regular AdamW with a cosine schedule. We found some benefit from AdEMAMix (Pagliardini et al., 2025) in small-scale runs compared to AdamW, but it performed worse than Muon, and a combination of both did not seem to help. For AdamW, decreasing _β_ 2 showed improvements at least for shorter runs. The comparisons between cautious weight decay, weight decay, and no weight decay were not very clear; we went with cautious weight decay due to its inclusion in the NanoGPT speedrun (Jordan et al., 2024a). Since Ma et al. (2025a) used label smoothing but this can hurt the performance on metrics like logloss, we tried a label smoothing schedule that decays to zero at the end of training, but it resulted in equal performance. 

**Architecture: embeddings.** We did not see benefits from using MLPs instead of linear layers for embedding _x_ . Adding log( _n_ ) together with _x_ was not helpful in small runs either. Surprisingly, using regular column-wise attention instead of ISAB performed worse in some runs. Mixing the layers of the column- and row-attention stages did not seem beneficial, and it can be a disadvantage since it requires more transposes and yields a less optimized CPU- and disk-offloading. Even with such mixing, it did seem worse to have a separate column for _y_ instead of adding the embedding of _y_ to the embeddings of the columns _xi_ . 

**Architecture: row interaction.** It seems that the full row-wise attention (attention across columns, within a row) is important, replacing it by induced self-attention performed considerably worse. We experimented a bit with random feature identifies in the version used by LimiX (Zhang et al., 2025b), but it was unclear if they are beneficial. 

**Architecture: normalizations.** We did not see improvements from different placements of normalization layers (though the experiments used more shallow nets). Additionally, bias-free or parameter-free layernorms seemed to perform similar to full layernorms while being faster, but we were not very confident in whether these measurements are good enough. 

**Architecture: other.** TabPFNv2-type architectures seemed to perform well, and we have no clear conclusion in which situations they are better or worse than the TabICLv2 architecture. Due to the higher runtime complexity and per-step time of the TabPFNv2 architecture, we discarded it. Experiments with residual connections did not show much differences in the 

20 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
0 . 80<br>0 . 75<br>0 . 70<br>0 . 65<br>0 . 60<br>TabICL RealTabPFN-2.5<br>0 . 55 TabPFNv2 TabICLv2<br>0 50k 100k 150k 200k 250k<br>Pretraining Step<br>(a)  Ablation results based on normalized accuracy<br>1200<br>1100<br>1000<br>900<br>800<br>TabICL RealTabPFN-2.5<br>700<br>TabPFNv2 TabICLv2<br>600<br>0 50k 100k 150k 200k 250k<br>Pretraining Step<br>(b)  Ablation results based on Elo<br>TabICL RealTabPFN-2.5<br>0 . 4<br>TabPFNv2 TabICLv2<br>0 . 3<br>0 . 2<br>0 . 1<br>0 50k 100k 150k 200k 250k<br>Pretraining Step<br>Ref with TabICL prior Ref  − feature grouping<br>Ref with TabICL architecture Ref + prior noise<br>Ref  − early target Ref (2 runs)<br>Ref : Muon  → AdamW Ref + more layers<br>Ref  − prior filtering Ref + QASSMax<br>Accuracy<br>Min-Max Norm.<br>Elo<br>Log-Loss<br>Min-Max Norm.<br><!-- End of picture text -->



<!-- Start of picture text -->
(c)  Ablation results based on log-loss<br><!-- End of picture text -->

_Figure C.1._ **Ablation results using different metrics.** 

21 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

results. Increasing the number of attention heads from 4 to 8 in TFcol and TFicl seemed slightly beneficial in small-scale runs with a smaller model, but not necessarily in large-scale runs. 

**Other things.** For regression (judged by MSE), we found incremental spline quantile functions (ISQF, Park et al., 2022) to work similarly well as regular quantile regression, but discarded them due to their computational (and conceptual) overhead. While LimiX (Zhang et al., 2025b) included convolution-based functions in their prior, we did not see a measurable benefit from including these functions into our prior, at least when fine-tuning TabICL on our prior. TabICL uses a masking mechanism to deal with micro-batches in which some datasets have fewer columns, where the rest of columns are filled up with zeros. We did not implement this and the pretraining still worked well. 

### **E. Details on the prior** 

In the following, we describe left-out details from the prior description in Section 5, with a focus on a more implementationrelated description. To keep the prior modular, we decompose it into sampling methods for different objects: random datasets, random functions, random points, random matrices, and so on. These will be described in the following subsections (which may sometimes refer to components introduced by later subsections). Some of the components are visualized in Appendix F. 

#### **E.1. Differences to previous priors** 

The closest prior to ours is probably the TabPFNv2 prior (Hollmann et al., 2025), though not all details about it are known. However, our prior still differs from it in many ways by introducing new mechanisms: 

- We introduce a new correlated scalar sampling mechanism (Appendix E.2). 

- We introduce a new random graph sampling mechanism (Appendix E.4). 

- We introduce additional computations at each node that create random node and feature importances (Appendix E.5). 

- We make the extraction of numerical and categorical features more precise through the introduction of random converters (Appendix E.6) and provide more variants for categorical converters. 

- We explicitly introduce multiple ways to apply random functions in the case of multiple parent nodes (Appendix E.7, it is unclear if this case can occur in TabPFNv2). 

- We introduce new random function types and diversify existing ones (Appendix E.8). In particular, for tree-based functions we do not only use single trees like TabPFNv2. Instead, we use an ensemble of CatBoost-style symmetric trees which facilitate efficient computations. Our GP functions are extended and multivariate versions of the random GP activations from TabICL and come with a detailed theoretical analysis (Appendix G). We also introduce random linear, quadratic, clustering-based, and product functions. 

- We add more random activations (Appendix E.9). 

- While TabPFNv2 uses random Gaussian matrices, we introduce four additional random matrix types (Appendix E.10). 

- We introduce random weights that are useful in multiple places, be it for sampling correlated categoricals, feature importances, or singular values (Appendix E.11). 

- We introduce more mechanisms to sample random points, by applying a random function to different kinds of base distributions (Appendix E.12). 

- We introduce a filtering mechanism similar to Dong et al. (2025) in Appendix E.14. 

- In addition, for scalar random variables like the number of categories etc., we generally use different choices of distributions, either to fit our framework, or because they are not known for TabPFNv2. 

22 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **E.2. Sampling correlated scalars** 

When sampling scalar numerical or categorical values within the prior, we want some of them to be correlated, since we expect some properties to be correlated in real datasets. For example, correlated quantities could include the cardinalities of different categorical columns, or the random function types at different nodes of the graph. To know which values should be correlated, we assign names to them such as “categorical ~~c~~ ardinality”. All values with that name are sampled from the same distribution, whose parameters are themselves sampled once for every name. 

**Numerical values:** For each variable name, we sample _t ∼_ Uniform[0 _,_ 1] and _s ∼_ LogUniform[0 _._ 1 _,_ 10000] and set _α_ := _st, β_ := _s_ (1 _− t_ ). For each time a variable should be sampled for that name, a base variable is sampled as _u ∼_ Beta( _α, β_ ). Then, it is affinely transformed to a desired range, the exponential is taken for log-type distributions, and it is rounded down for integer-valued random variables. We denote uniform-like real-valued and integer-valued random variables with bounds _a, b_ by Num( _a, b_ ) _,_ Int( _a, b_ ) and log-uniform-like versions as LogNum( _a, b_ ) _,_ LogInt( _a, b_ ). 

**Categorical values:** For each variable name, we generate a random weights vector _w ∈_ R<sup>_c_</sup> , where _c_ is the desired number of categories. For each time a variable should be sampled from that name, it is sampled from the distribution represented by the normalized weight vector _w_ . Note that the generation of the random weights vector _w_ uses the numerical sampling mechanism for some scalar parameters, so the random weights vectors for different variable names are correlated. 

#### **E.3. Random dataset** 

We first sample some general characteristics of the dataset. For classification, we sample the number of classes from UniformInt(2 _,_ 10). The ratio of categorical columns is sampled from Uniform( _−_ 0 _._ 5 _,_ 1 _._ 2) and subsequently clipped to [0 _,_ 1]. The categorical cardinalities are sampled from LogInt(2 _, M_ ), where the maximum cardinality is sampled once as _M ∼_ LogInt(2 _,_ 9), and a uniform random fraction of them is sampled through the correlated sampling mechanism. The total number of columns is configurable but is sampled from UniformInt(2 _,_ 100) in our training runs. 

We then sample a random graph with LogInt(2 _,_ 32) nodes. To assign the different target columns to nodes in the randomly sampled graph, for each column type (either input columns for _x_ , or the target column _y_ ), we sample the number of eligible nodes uniformly, then sample a subset of nodes of that size, and then assign each column to a random node in that subset. The graph sampling and node assignment is potentially repeated until the graph filtering mechanism accepts it. The nodes are then traversed in topological order and their corresponding random node functions are called with the data from all parent nodes as input. Nodes that are not needed for the final dataset computation are pruned. Finally, the obtained dataset is shuffled randomly and split into train and test parts. 

#### **E.4. Random graph** 

For node indieces _i < j_ , an edge is placed with probability 



where _A, Bi, Cj_ are independent standard Cauchy random variables. _A_ controls the general level of connectivity, while _Bi_ and _Cj_ control the individual outgoing and incoming connectivities of the nodes, respectively. Compared to independent Bernoulli probabilities, this model yields more diverse connectivity patterns. Cauchy random variables have heavy tails and therefore yield higher probabilities of “exceptions to the rule”. 

#### **E.5. Random node function** 

We first obtain a matrix _X ∈_ R<sup>_n×di_</sup> from a random multi-function applied to the parent node data, or from the random points mechanism if there are no parents. Here, _di_ =<sup>�</sup> _j_<sup>_dij_+ LogInt(1</sup><sup>_,_32), where the</sup><sup>_dj_are the dimensions required</sup> by the random converters for extracting the dataset columns from the node. A random converter can extract a column and also modify (e.g., discretize) the corresponding portion of the node data. We then standardize the columns of _X_ . Then, we sample a weights vector _w ∈_ R<sup>_di_</sup> and multiply each column of _X_ by the corresponding weight. Afterwards, we divide _X_ by the average (over samples) _L_<sup>2</sup> norm of the vectors. The motivation to use the _L_<sup>2</sup> norm instead of the RMS norm is to keep the vector norms small in high dimensions, such that high-dimensional functions do not become too difficult to learn. Now, we apply the random converters for the assigned columns to their respective _dij_ dimensions, updating the respective part of _X_ with their output. Finally, in the “random rescale” step, we multiply _X_ by a scalar LogNum(0 _._ 1 _,_ 10). 

23 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **E.6. Random converter** 

We introduce converters 

_X_<sup>_′_</sup> _, v_ = converter( _X_ ) _, X, X_<sup>_′_</sup> _∈_ R<sup>_n×d_</sup> _, v ∈_ R<sup>_n_</sup> _,_ 

which extract a column _v_ for the generated dataset while also potentially modifying the node data _X_ to _X_<sup>_′_</sup> . 

**Numerical converters:** We set _v_ = _X_ , _d_ = 1, and choose _X_<sup>_′_</sup> = _f_ ( _X_ ), where _f_ is sampled as the identity or a warping function based on a Kumaraswamy distribution (Kumaraswamy, 1980) after min-max scaling, following Hollmann et al. (2025). For the Kumaraswamy warping, we min-max scale values _x_ to the range [0 _,_ 1] and then compute 1 _−_ (1 _− x_<sup>_a_</sup> )<sup>_b_</sup> with _a, b ∼_ LogNum(0 _._ 2 _,_ 5). The Kumaraswamy warping is unintentionally applied to compute _x_<sup>_′_</sup> instead of _v_ . 

**Categorical converters:** Let _c ∈_ N be the number of desired categories. We use two main approaches: 

- **Neighbor-based** : We choose a random subset of _c_ points from the data. As in the RandomDiscretizationFunction, we then map each point _x_ to its closest point in the subset as measured by the _L_<sup>_p_</sup> distance, _p_ = LogNum(0 _._ 5 _,_ 4). The index of the closest center is the class index. For neighbor-based approaches, we sample the desired dimension _d_ of _x_ as _c_ with probability 1 _/_ 2 and Int(1 _, c −_ 1) otherwise. 

- **Softmax-based** : We sample the category from 

#### softmax( _ax_ ˜ + _b_ ) _,_ 

where _x_ ˜ is a standardized version of the input _x ∈_ R<sup>_c_</sup> , _a ∼_ LogNum(0 _._ 1 _,_ 10), and _b_ = log( _w_ + 10<sup>_−_4</sup> ) with _w ∈_ R<sup>_c_</sup> being a random weight vector. The variation in _a_ can create different levels of separation between categories, and the variation in _b_ can create different levels of imbalance. For softmax-based approaches, we always need the dimension of _x_ to be _d_ = _c_ . 

We further distinguish different approaches to compute the transformed node vector _x_<sup>_′_</sup> : 

- Output the input _x_ (neighbor- or softmax-based) 

- Output the category index _i_ , repeated to get a _d_ -dimensional vector (neighbor- or softmax-based). 

- Output the closest center (neighbor-based) or a random function applied to the closest center (neighbor-based). 

- Sample random points _{z_ 1 _, . . . , zc}_ , then output _zi_ where _i_ is the category index (softmax-based). 

In total, we obtain seven combinations of categorical converters, of which we sample one randomly. 

#### **E.7. Random multi-function** 

If there is only a single input node, we use a random function (see below). Otherwise, if there are _n_ in input nodes, we proceed as follows: With probability 1 _/_ 2, we concatenate the tensors of all input nodes along the features dimension and apply a random function to it. Else, we apply separate random functions to each input node, obtaining _n_ in tensors of dimension _n × d_ that are aggregated along the _n_ in axis using one of the following four element-wise aggregations: sum, product, max, or logsumexp. 

#### **E.8. Random functions** 

**RandomNNFunction** A random NN with LogInt(1 _,_ 3) linear layers, hidden width LogInt(1 _,_ 127) (drawn once per NN), and a 50% chance each of including an activation at the beginning or end of the network. The linear layers use RandomLinearFunction (no bias, but there is a bias in the activation function). 

**RandomTreeFunction** Generates LogInt(1 _,_ 128) trees, of depth Int(1 _,_ 7). Each tree is symmetric (= oblivious), meaning that it uses the same splitting criterion for all nodes on the same layer. The split dimension is chosen randomly with probability proportionally to the standard deviation of data in that dimension, to respect the feature importances that were randomly sampled on the input nodes. The split points are random samples from the arriving data in the respective dimension. The leaf values are standard normal random values. Then, each tree is evaluated for the data and the corresponding leaf values are averaged. 

24 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

**RandomDiscretizationFunction** Chooses a subset of samples from _X_ as centers, with the number of samples being LogInt(2 _,_ 255). It then maps each point _x_ to its closest center as measured by the _L_<sup>_p_</sup> distance, _p_ = LogNum(0 _._ 5 _,_ 4), and applies a random linear function to the result. 

**RandomGPFunction** This function computes _f_ ( _Mx_ ) = ( _f_ 1( _Mx_ ) _, . . . , fd_ ( _Mx_ )), where each component _fi_ is sampled from a Gaussian process with a random kernel _k_ shared between all _i_ . Here, _Mij_ = _αwiAij_ with random weights vector _w_ , random scale _α ∼_ LogNum(0 _._ 5 _,_ 10), and a random Gaussian matrix _A_ . This choice is inspired by the random GP activations in TabICL as well as the success of tuning over different kernels with different bandwidths (scales) and learnable linear input transformations in xRFM (Beaglehole et al., 2025). 

The first question is how to design the random kernels _k_ . In order to use a random Fourier features approximation (Rahimi & Recht, 2007), we design _k_ directly in the Fourier domain. Suppose that _g_ : R<sup>_d_</sup> _→_ R _≥_ 0 is integrable and even ( _g_ ( _x_ ) = _g_ ( _−x_ ) ˇ for all _x_ ). Then _k_ ( _x, x_<sup>_′_</sup> ) = _g_ ( _x − x_<sup>_′_</sup> ) is a real-valued kernel on R<sup>_d_</sup> (see e.g. Theorem G.2), where ˇ _g_ ( _x_ ) = � _e_<sup>_i⟨x,ω⟩_</sup> _g_ ( _ω_ ) _dω_ is the inverse Fourier transform of _g_ . In Appendix G, we show that the tail behavior of _g_ is directly related to the smoothness of functions sampled from the GP: If there exist constants _c, C, r_ 0 _>_ 0 such that 



then the sample paths (sampled functions) from the GP essentially have smoothness ( _q − d_ ) _/_ 2, at least when _q >_ 2 _d_ . If _g_ is a probability density function (integrates to 1), we can approximate _k_ using random Fourier features (Rahimi & Recht, 2007): 



where the rows of _W_ are independently drawn from _g_ and _bi ∼_ Unif[0 _,_ 2 _π_ ] i.i.d. The dimension _p_ can be chosen arbitrarily large to improve the approximation quality; we follow Qu et al. (2025) and set _p_ = 256. 

As argued in Qu et al. (2025), for a standard normal random vector _z ∈_ R<sup>_p_</sup> , _z_<sup>_⊤_</sup> _ϕ_ ( _x_ ) follows a Gaussian process with kernel _ϕ_ ( _x_ )<sup>_⊤_</sup> _ϕ_ ( _x_<sup>_′_</sup> ), which therefore approximates the Gaussian process with kernel _k_ . For a general multi-output setting, we therefore sample _Z ∈_ R<sup>_d_out</sup><sup>_×p_</sup> with i.i.d. standard normal entries and compute (omitting the factor _√_ 2 which only rescales the output) 



with _w, A_ from the beginning of the explanation. 

It remains to find a probability density function _g_ for a given _q_ from which we can efficiently sample. Since we can choose a rotation-invariant distribution, we can just sample it as _ω_ = _rz/∥z∥_ , where _z ∈_ R<sup>_d_</sup> is a standard normal vector and _r_ controls the radial component of _ω_ . For densities with tail Θ( _∥ω∥_<sup>_−q_</sup> ), integration in spherical coordinates yields that the tail of the density of _r_ = _∥ω∥_ must behave like Θ( _r_<sup>_d−_1</sup> _r_<sup>_−q_</sup> ) = Θ( _r_<sup>_d−_1</sup><sup>_−q_</sup> ). 

We construct a family of 1D distributions with power-law tails that are easy to sample from: For _a >_ 1, we define a CDF of _Ha_ ( _r_ ) = 1 _−_ (1 + _r_ )<sup>1</sup><sup>_−a_</sup> (for _r ≥_ 0). The associated PDF is _ha_ ( _r_ ) = _Ha_<sup>_′_(</sup><sup>_r_)=(</sup><sup>_a −_1)(1 +</sup><sup>_r_)</sup><sup>_−a_.We can then sample</sup> from _H_ using inverse CDF sampling: For _u ∼_ Unif[0 _,_ 1], 



is distributed according to _Ha_ . We sample _a ∼_ LogNum(2 _,_ 20), corresponding to _q_ = _a_ + _d −_ 1, and sample _r_ from _Ha_ . 

With 50% probability, we choose **another way** to sample the kernel: Inspired by the choice of non-rotationally invariant axis-aligned kernels in xRFM (Beaglehole et al., 2025), we alternatively sample each entry of _ω_ independently from _Ha_ . This yields a product distribution _g_ ( _ω_ ) = _g_ 1( _ω_ 1) _· . . . · gd_ ( _ωd_ ) for _ω_ , whose inverse Fourier transform ˇ _g_ yields a kernel that is a product of one-dimensional kernels, as for the case “ _p_ = _q_ ” in xRFM (Beaglehole et al., 2025). We do not explicitly prove a result about the path smoothness for product kernels, but a similar argument to Theorem G.1 in conjunction with Theorem 4.2 of Steinwart (2024) should yield that for _a >_ 2, the paths are contained in Sobolev spaces of dominating mixed smoothness of order _s_ whenever _s <_ ( _a −_ 1) _/_ 2. As in the other case, we sample _a ∼_ LogNum(2 _,_ 20). Since the product kernel constructed in this way is axis-aligned, we do not apply the random matrix _M_ in the construction above to preserve axis-alignment. 

**RandomLinearFunction** Samples a random matrix (Appendix E.10) and multiplies each vector _x_ by this matrix. 

25 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

**RandomQuadraticFunction** Computes 



where each _Mi_ is a random matrix, jointly sampled from the same random matrix type. To avoid quadratic complexity in the dimension of _x_ , we first subsample _x_ to at most 20 dimensions if it has more than 20 dimensions. We include linear and constant terms by appending 1 to the vector _x_ . 

**RandomEMAssignmentFunction** This function type is inspired by the computation of the probability _pi_ that the input _x_ is from the cluster _i_ in the EM algorithm. However, we add some things like different _L_<sup>_p_</sup> -norms and random powers _q_ without making sure that this corresponds to any real “cluster assignment” computation, simply for further increasing the diversity of computed functions. 

First, a number _m_ = LogInt(2 _,_ max _{_ 16 _,_ 2 _d_ out _}_ ) of “Gaussians” is sampled. Then, centers _µ_ 1 _, . . . , µm_ are chosen using random input vectors plus standard normal noise. Standard deviations _σ_ 1 _, . . . , σm_ are chosen independently as exp(0 _._ 1 _·_ Normal(0 _,_ 1)). Then, logits are computed as 



where _p_ = LogNum(1 _,_ 4), _q_ = LogNum(1 _,_ 2). (The Gaussian case would be using _p_ = 2, _q_ = 1, and using _d/_ 2 instead of 1 _/_ 2 in the equation.) Finally, the output is computed as 



where _L_ is a random linear function. 

**RandomProductFunction** Computes _f_ ( _x_ ) _g_ ( _x_ ), where _f, g_ are two random functions (not product, NN, or EM, to optimize speed). 

#### **E.9. Random activations** 

Following TabICL and TabPFNv2, we further expand the set of available activation functions. By activation functions, we mean functions _f_ : R<sup>_d_</sup> _→_ R<sup>_d_</sup> that preserve the input dimension, even if they are not element-wise. 

We use them as follows inside a NN, expanding upon the standardization + random rescaling from TabICL: 

- We first standardize (subtract the mean and divide by the standard deviation) along the batch dimension. 

- We rescale randomly, as _x ← a_ ( _x − b_ ), where _a_ = LogNum(1 _,_ 10) and _b_ is a random sample from the standardized data. We choose _b_ this way to avoid getting only zeros for activations like ReLU that are zero in a large portion of the space. 

- Now, we apply the activation. 

- Finally, we standardize again. 

For the activation, with probability 2 _/_ 3 we pick one of the following fixed activations, otherwise one of the parametric activations below that. 

**Fixed activations.** As fixed activations, we use Tanh, LeakyReLU, Elu, Identity, SELU, SiLU, ReLU, softplus, ReLU6, HardTanh, signum, Heaviside, exp( _−x_<sup>2</sup> ), exp, 1[0 _,_ 1], sin, square, abs, softmax, one-hot argmax, argsort, logsigmoid, log(max( _|x|,_ 10<sup>_−_6</sup> )), rank, sigmoid, round, modulo 1. 

**Parametric activations.** We introduce the following activations with random parameters: 

- ReLU( _x_ )<sup>_q_</sup> with _q ∼_ LogNum(0 _._ 1 _,_ 10). 

26 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

- sign( _x_ ) _|x|_<sup>_q_</sup> with _q ∼_ LogNum(0 _._ 1 _,_ 10). 

- ( _|x|_ + 10<sup>_−_3</sup> )<sup>_−q_</sup> with _q ∼_ LogNum(0 _._ 1 _,_ 10). 

- _x_<sup>_m_</sup> with _m ∼_ Int(2 _,_ 5). 

#### **E.10. Random matrix** 

We randomly sample a matrix from one of the following five types: 

RandomGaussianMatrix: Consists of i.i.d. standard normal entries. 

RandomWeightsMatrix: To sample a matrix of shape _m × k_ , we compute _Mij_ = _Wij ⊙ Gij_ , where _G_ is a random Gaussian matrix and _Wi,·_ are random weight vectors (which are in general correlated through the correlated sampling mechanism). Afterwards, the rows of _M_ are normalized (divided by their norm). 

RandomSingularValuesMatrix: To sample a matrix of shape _m × k_ , we compute _U_ diag( _w_ ) _V_<sup>_⊤_</sup> , where _w ∈_ R<sup>min</sup><sup>_{m,k}_</sup> is a random weights vector and _U, V_ are random Gaussian matrices of suitable shape. While sampling orthogonal _U, V_ would mean that we explicitly sample the singular value decomposition, using Gaussian _U, V_ is faster and still produces a rotation-invariant distribution (since for Gaussian _U_ and arbitrary orthogonal matrix _R_ , the distribution of _U_ is the same as the distribution of _UR_ or _RU_ ). 

RandomKernelMatrix: To sample a matrix of shape _m × k_ , we sample _k_ + _m_ random covariance points _x_ 1 _, . . . , xk_ + _m ∈_ R<sup>3</sup> and a scaling factor _γ ∼_ LogNum(0 _._ 1 _,_ 10), then create the Laplace kernel matrix _Kij_ = exp( _−γ∥xi − xm_ + _j∥_ ) and multiply each entry by an independent random sign (a number in _{−_ 1 _,_ 1 _}_ ). 

RandomActivationMatrix: After sampling a matrix from one of the other types, we apply a random activation to the flattened matrix, then add Gaussian noise with standard deviation 10<sup>_−_3</sup> . Unlike for the NN, we omit the standardization and random rescaling in the activations. 

**Postprocessing.** After creating a matrix using one of the described types, we add 1e-6 times a random Gaussian matrix and normalize each row of the resulting matrix. This prevents all-zero rows that could arise from some activation functions in the RandomActivationMatrix. 

#### **E.11. Random weights** 

To emulate random feature importances, singular value decays, or (unnormalized) probability distributions, we introduce a dedicated way to sample random positive vectors _w ∈_ R<sup>_d_</sup> _>_ 0<sup>.We first generate</sup> 



with _q ∼_ LogNum(0 _._ 1 _/_ log( _d_ + 1) _,_ 6) and _σ ∼_ LogNum(10<sup>_−_4</sup> _,_ 10). The lower and upper bounds for _q_ are chosen such that we can sample vectors where no weights are close to zero as well as vectors where almost all weights are close to zero. Finally, we normalize and shuffle _w_ . 

#### **E.12. Random points** 

Generating a matrix _X ∈_ R<sup>_n×d_</sup> of random points is in principle the same problem as generating a random dataset with numerical columns, but here we only use cheaper mechanisms (and avoid infinite recursions). First, we sample either standard normal points, uniform on [ _−_ 1 _,_ 1]<sup>_d_</sup> , uniform on the unit ball, or normal with random covariance. Then, we apply a random function to these points. The normal points with random covariance are sampled as follows: For a given dimension _d_ , sample _x ∈_ R<sup>_d_</sup> from a standard normal distribution, then compute _A_ ( _w ⊙ x_ ), where _⊙_ is the elementwise product, _w ∈_ R<sup>_d_</sup> are random weights, and _A_ is a random Gaussian matrix. 

#### **E.13. Postprocessing** 

Following TabICL, columns with a single value are removed. Datasets are discarded if all columns were removed or less than 2 classes are present or the train and test splits cannot be fixed to contain the same classes. We ordinal-encode categoricals. For all columns _xi_ and in the regression case also _y_ , we remove outliers, then standard-scale. We permute the column order and the class indices (but not the categorical indices). 

27 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **E.14. Filtering** 

The ExtraTrees-based filtering works as follows: We convert classification problems to regression using one-hot encoding, to unify the two cases. To obtain very fast filtering, we then fit an ExtraTreesRegressor (Geurts et al., 2006) from scikit-learn (Pedregosa et al., 2011) with n estimators=25, bootstrap=True, and max ~~d~~ epth=6 on the full dataset. We then test whether the out-of-bag predictions can achieve a lower MSE than the mean label by checking if this is the case on at least 95% out of 200 bootstrap subsamples. If it is not the case, the dataset is rejected. 

### **F. Plots for the prior** 

Figure F.1 shows random datasets from the prior. Random function types are shown in Figures F.2, F.3, F.4, F.7, F.5, F.6, F.8, and F.9. Random graphs are shown in Figure F.10. Figure F.11 shows random points. Random matrices are visualized in Figures F.12, F.13, F.14, F.15, F.16. 



_Figure F.1._ **Random classification datasets from the prior.** Datasets contain 500 samples and two columns in _x_ . The color shows the class label. Only datasets containing at least 10 unique values on each axis are shown (for visualization purposes, since otherwise the points can overlap a lot). 

28 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



_Figure F.2._ **Samples of RandomNNFunction.** We use inputs from [ _−_ 3 _,_ 3]<sup>2</sup> and one-dimensional output. 



_Figure F.3._ **Samples of RandomTreeFunction.** We use inputs from [ _−_ 3 _,_ 3]<sup>2</sup> and one-dimensional output. 



_Figure F.4._ **Samples of RandomDiscretizationFunction.** We use inputs from [ _−_ 3 _,_ 3]<sup>2</sup> and one-dimensional output. 



_Figure F.5._ **Samples of RandomLinearFunction.** We use inputs from [ _−_ 3 _,_ 3]<sup>2</sup> and one-dimensional output. 

29 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



_Figure F.6._ **Samples of RandomQuadraticFunction.** We use inputs from [ _−_ 3 _,_ 3]<sup>2</sup> and one-dimensional output. 



_Figure F.7._ **Samples of RandomGPFunction.** We use inputs from [ _−_ 3 _,_ 3]<sup>2</sup> and one-dimensional output. 



_Figure F.8._ **Samples of RandomEMAssignmentFunction.** We use inputs from [ _−_ 3 _,_ 3]<sup>2</sup> and one-dimensional output. 



_Figure F.9._ **Samples of RandomProductFunction.** We use inputs from [ _−_ 3 _,_ 3]<sup>2</sup> and one-dimensional output. 

30 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



_Figure F.10._ **Randomly sampled graphs.** Graphs are not filtered (this would require knowing the assignment of columns to nodes). 

31 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



_Figure F.11._ **Samples of RandomPoints from the prior.** We sample 300 three-dimensional points and encode the third dimension through the color. 



_Figure F.12._ **Samples of RandomGaussianMatrix.** We sample 30 _×_ 30 matrices and show the absolute values of their entries. We permute their indices by sorting the absolute values of the top left- and right-singular vectors of their absolute values, respectively. 

32 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



_Figure F.13._ **Samples of RandomWeightsMatrix.** We sample 30 _×_ 30 matrices and show the absolute values of their entries. We permute their indices by sorting the absolute values of the top left- and right-singular vectors of their absolute values, respectively. 



_Figure F.14._ **Samples of RandomSingularValuesMatrix.** We sample 30 _×_ 30 matrices and show the absolute values of their entries. We permute their indices by sorting the absolute values of the top left- and right-singular vectors of their absolute values, respectively. 



_Figure F.15._ **Samples of RandomKernelMatrix.** We sample 30 _×_ 30 matrices and show the absolute values of their entries. We permute their indices by sorting the absolute values of the top left- and right-singular vectors of their absolute values, respectively. 



_Figure F.16._ **Samples of RandomActivationMatrix.** We sample 30 _×_ 30 matrices and show the absolute values of their entries. We permute their indices by sorting the absolute values of the top left- and right-singular vectors of their absolute values, respectively. 

33 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

### **G. Path smoothness for Gaussian processes** 

In the following, we prove a result for the smoothness of functions sampled from Gaussian processes. We quantify the smoothness in terms of which Sobolev spaces _H_<sup>_s_</sup> ( _B_ ) the functions _f_ belong to, for some domain _B ⊆_ R<sup>_d_</sup> and smoothness _s ≥_ 0. There are different ways to define Sobolev spaces, and we refer the curious reader to the literature, but essentially functions in _H_<sup>_s_</sup> ( _B_ ) have an _s_ -th derivative that is square-integrable. For non-integer _s_ , the Sobolev-Slobodeckij norm treats the fractional part _s −⌊s⌋_ using a H¨older-like criterion on the _⌊s⌋_ -th derivative. 

**Notation.** We write GP(0 _, k_ ) for the distribution of Gaussian processes with mean zero and covariance kernel _k_ . We assume that such Gaussian processes are defined on a complete probability space. We say that two stochastic processes ( _Xt_ ) _t∈T ,_ ( _Yt_ ) _t∈T_ on the same probability space (Ω _, F, P_ ) are a modification (or version) of each other if _P_ ( _Xt_ = _Yt_ ) = 1 for all _t_ . Moreover, for integrable _g_ : R<sup>_d_</sup> _→_ R _≥_ 0, we denote its inverse Fourier transform as ˇ _g_ ( _x_ ) = � _e_<sup>_i⟨ω,x⟩_</sup> _g_ ( _ω_ ) _dω_ . 

In the following theorem, it might be possible to relax the criterion _q >_ 2 _d_ to _q > d_ by using other results from the literature. However, in this case the result would not guarantee anymore that the paths are continuous. 

**Theorem G.1** (Smoothness of GP sample paths) **.** _Let g_ : R<sup>_d_</sup> _→_ R _≥_ 0 _be integrable and even such that there exist constants c, C, r_ 0 _>_ 0 _and q >_ 2 _d with_ 



_Then, k_ : R<sup>_d_</sup> _×_ R<sup>_d_</sup> _→_ R _, k_ ( _x, x_<sup>_′_</sup> ) := _g_ ˇ( _x − x_<sup>_′_</sup> ) _is a kernel._ 

_Let X ∼_ GP(0 _, k_ ) _, let B ⊆_ R<sup>_d_</sup> _be an arbitrary ball, and set s∗_ := ( _q − d_ ) _/_ 2 _. Then,_ 

- _For every s < s∗, there exists a modification Y of X such that P_ ( _Y |B ∈ H_<sup>_s_</sup> ( _B_ )) = 1 _._ 

- _For every s ≥ s∗ and every modification Y of X, P_ ( _Y |B ∈ H_<sup>_s_</sup> ( _B_ )) = 0 _._ 

_Proof._ **Step 1: “Repair” the spectral density in the center.** It follows from Theorem G.2 that _k_ is a kernel. To apply the remainder of Theorem G.2, we need a spectral density _g∗_ satisfying 



for suitable constants _c∗, C_<sup>_∗_</sup> _>_ 0 and all _ω ∈_ R<sup>_d_</sup> , not just those with large enough radius. Our function _g_ may not satisfy this since Eq. (G.1) only needs to hold for large enough _∥ω∥_ . Using the ball _Br_ 0 := _{ω ∈_ R<sup>_d_</sup> _| ∥ω∥≤ r_ 0 _}_ , define 



Then, _g_ = _g_ hi + _g_ lo, and we can set _g∗_ = _g_ hi + _g_ 0. Since all of these functions are non-negative, integrable, and even, we can also consider their associated kernels. If _X_ hi _∼_ GP(0 _, k_ hi) _, X_ lo _∼_ GP(0 _, k_ lo) _, X_ 0 _∼_ GP(0 _, k_ 0) are independent, then _X_ := _X_ hi + _X_ lo _∼_ GP(0 _, k_ ) and _X∗_ := _X_ hi + _X_ 0 _∼_ GP(0 _, k∗_ ). 

**Step 2: Equivalence to repaired version.** The inverse Fourier transforms ˇ _g_ lo _,_ ˇ _g_ 0 are infinitely differentiable: for example, since _g_ lo is supported on a ball of radius _r_ 0, we have 



where the integration and differentiation can be exchanged because 



Hence, the associated kernels _k_ lo _, k_ 0 are infinitely differentiable. For a given smoothness _s ∈_ R, using Theorem 4.1 of Da Costa et al. (2023) and choosing a suitable modification of _X_ lo _, X_ 0, we almost surely have _X_ lo _|B, X_ 0 _|B ∈ H_<sup>_s_</sup> ( _B_ ). Hence, if there exists a modification _X_<sup>˜</sup> of _X_ with _P_ ( _X_<sup>˜</sup> _|B ∈ H_<sup>_s_</sup> ( _B_ )) = _p_ for some _p ∈_ [0 _,_ 1], then _X_<sup>˜</sup> _∗_ := _X_<sup>˜</sup> _− X_ lo + _X_ 0 

34 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

is a modification of _X∗_ with _P_ ( _X_<sup>˜</sup> _∗|B ∈ H_<sup>_s_</sup> ( _B_ )) = _p_ . The same holds with switched roles of _X_ and _X∗_ . Hence, it suffices to prove the desired smoothness results for _X∗_ instead of _X_ . 

**Step 3: Smoothness of sample paths for** _X∗_ **.** By Theorem G.2 and Eq. (G.2), the RKHS _H∗_ of _k∗_ is equivalent to the Sobolev space _H_<sup>_q/_2</sup> (R<sup>_d_</sup> ), and therefore _H∗|B_ is equivalent to _H_<sup>_q/_2</sup> (R<sup>_d_</sup> ) _|B_ , which is equivalent to _H_<sup>_q/_2</sup> ( _B_ ). Hence, the inclusion operator _I_ : _H∗|B → H_<sup>_q/_2</sup> ( _B_ ) is bounded and so is its inverse. For Hilbert spaces _A ⊆ B_ , Steinwart (2024) defines in Definition 1.1 that _A ≪ B_ if the associated inclusion operator from _A → B_ is Hilbert-Schmidt. From Lemma 6.9 of Steinwart (2024), if _s > d/_ 2, then _H_<sup>_q/_2</sup> ( _B_ ) _≪ H_<sup>_s_</sup> ( _B_ ) if and only if _s <_ ( _q − d_ ) _/_ 2 = _s∗_ . By Theorem 15 of Bell (2016), since _I, I_<sup>_−_1</sup> are bounded, we obtain _H∗|B ≪ H_<sup>_s_</sup> ( _B_ ) if and only if _s <_ ( _q − d_ ) _/_ 2. By Theorem 1.2 in Steinwart (2024) (which is a reformulation of a result of Lukic´ & Beder 2001), the smoothness criterion for _X∗_ is equivalent to _H∗|B ≪ H_<sup>_s_</sup> ( _B_ ), which completes the proof for the cases _s > d/_ 2. But since the spaces for _s ≤ d/_ 2 are supersets of those for _s > d/_ 2, the statement extends to them as well. 

**Lemma G.2** (Fourier characterization of Sobolev kernels) **.** _Let g_ : R<sup>_d_</sup> _→_ R _≥_ 0 _be even and integrable. Then, k_ : R<sup>_d_</sup> _×_ R<sup>_d_</sup> _→_ R _, k_ ( _x, y_ ) = _g_ ˇ( _x − y_ ) _∈_ R _is a kernel. Moreover, if there exist constants C >_ 0 _and s > d/_ 2 _with_ 



_for all ω ∈_ R<sup>_d_</sup> _, then the reproducing kernel Hilbert space (RKHS) Hk associated with k is equivalent to the Sobolev space H_<sup>_s_</sup> (R<sup>_d_</sup> ) _, meaning that they are equal as sets and their norms are equivalent._ 

_Proof._ **Step 1: Constructing the RKHS via a feature space.** Define _g_ 0 : R<sup>_d_</sup> _→_ R _≥_ 0 _, ω �→_ (1 + _∥ω∥_<sup>2</sup> )<sup>_−s_</sup> and _g_ 1 := _g_ . Since _s > d/_ 2, _g_ 0 is integrable, and _g_ 1 is integrable by assumption. We will define feature maps into _H_ := _L_<sup>2</sup> (R<sup>_d_</sup> _,_ C), the Hilbert space of complex-valued square-integrable functions on R<sup>_d_</sup> . For _i ∈{_ 0 _,_ 1 _}_ , define 



which is valid since _g ∈ L_<sup>1</sup> implies<sup>_√_</sup> _<u>g</u> ∈ L_<sup>2</sup> . This feature map is associated with the kernel 



which is real-valued since _gi_ is even. Especially, _k_ = _k_ 1 is a real-valued kernel. Now, by Theorem 4.21 in Steinwart (2008), the associated RKHS is 



with norm 



**Step 2: Equivalence of the RKHSs.** Now, by Eq. <u>(G.3), we have</u> _C_<sup>_−_1</sup> _g_ 0 _≤ g_ 1 _<u>≤</u> Cg_ <u>0.</u> If _f ∈H_ <u>0, then there exists</u> _h ∈ H_ with _f_ = _⟨h, ϕ_ 0( _·_ ) _⟩H_ , and therefore _f_ = _⟨h_ � _g_ 0 _/g_ 1 _, ϕ_ 1( _·_ ) _⟩H ∈H_ 1, since _∥h_ <u>�</u> _g_ 0 _/g_ 1 _∥H ≤ √C∥h∥H_ . Together with the norm characterization above it therefore follows that _H_ 0 _⊆H_ 1 with _∥f ∥H_ 1 _≤ √C∥f ∥H_ 0 . By switching the roles of _g_ 0 and _g_ 1, we therefore conclude that _H_ 0 and _H_ 1 are equivalent. 

**Step 3:** _H_ 0 **is the desired Sobolev space.** Eq. (G.4) yields _k_ 0( _x, y_ ) = � _e_<sup>_i⟨x−y,ω⟩_</sup> (1 + _∥ω∥_<sup>2</sup> )<sup>_−s_</sup> _dω_ , which is up to a constant factor exactly the kernel of (an equivalent version of) the Sobolev space _H_<sup>_s_</sup> (R<sup>_d_</sup> ) (see e.g. De Vito et al., 2021, section 7.1). By step 2, this means that the RKHS of _g_ is equivalent to _H_<sup>_s_</sup> (R<sup>_d_</sup> ). 

35 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

### **H. Inference optimization for TabICLv2** 

#### **H.1. Efficient attention computation via selective query-key-value projections** 

We implement two attention optimizations that reduce redundant computation by selectively computing query, key, and value projections based on the specific attention patterns required in each stage of TabICLv2. 

**Row-wise inter-feature interaction.** During the row-wise interaction, we prepend _c_ learnable [CLS] tokens (we use _C_ = 4 as TabICL) to the feature embeddings and use only their final outputs that concatenated into a single vector as the row representation for the subsequent in-context learning. Since only the [CLS] token outputs are required, we optimize the final block of TFrow by restricting the query computation to these _c_ positions while allowing them to attend to the full sequence ([CLS] tokens and all features) as keys and values. Concretely, given a sequence of length _c_ + _m_ , the final block computes attention outputs only for the first _C_ query positions: 



where **Q** 1: _c ∈_ R<sup>_c×d_</sup> represents queries from [CLS] tokens only, while **K** _,_ **V** _∈_ R<sup>(</sup><sup>_c_+</sup><sup>_m_)</sup><sup>_×d_</sup> span the full sequence. This reduces the query projection cost from _O_ (( _c_ + _m_ ) _· d_<sup>2</sup> ) to _O_ ( _c · d_<sup>2</sup> ) and the attention computation from _O_ (( _c_ + _m_ )<sup>2</sup> _· d_ ) to _O_ ( _c ·_ ( _c_ + _m_ ) _· d_ ) in the final block. 

**Dataset-wise in-context learning.** During the final in-context learning, test samples learn from training samples via cross-attention, where test queries attend only to training keys and values. Since test samples never serve as context for other samples, computing their key and value projections is unnecessary. We therefore compute key and value projections only for the _n_ train training samples, while computing queries for all _n_ train + _n_ test samples: 



This reduces the key and value projection costs from _O_ (( _n_ train + _n_ test) _· d_<sup>2</sup> ) to _O_ ( _n_ train _· d_<sup>2</sup> ) each. 

**Layer normalization reuse.** We adopted the pre-norm setting in the transformer block. To avoid redundant computation, we first apply layer normalization to the full input sequence, then perform slicing to extract the required subset for query, key, or value computation. This ensures that normalization statistics are computed once over the complete sequence, and the sliced representations remain properly normalized without requiring separate normalization passes for different subsets. 

#### **H.2. Offloading technique** 

**Batch size estimation.** To dynamically adjust the batch size of the three transformers based on available memory and avoid out-of-memory errors, following TabICL, TABICLV2 employs polynomial regression to estimate the inference peak GPU memory consumption: 



Compared to TabICL, we introduce query-aware scalable softmax in TFcol and TFicl, which requires re-evaluating the memory coefficients for these components. The updated coefficients are as follows (TFrow remains unchanged from TabICL): 





where the estimated memory is measured in megabytes (MB). 

TabICLv2 demonstrates substantially improved capability for handling large tables compared to TabICL. To provide an affordable setup for processing tables with millions of samples, we implement a hierarchical offloading strategy that extends beyond the CPU offloading of TabICL to include disk-based offloading. 

36 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

**Memory bottleneck analysis.** Given an input _X ∈_ R<sup>_b×n×m_</sup> , where _b_ , _n_ , and _m_ represent the number of datasets, the number of samples, and the number of features respectively, _X_ is first reshaped to R<sup>(</sup><sup>_b×m_)</sup><sup>_×n_</sup> and processed by TFcol to produce feature embeddings _E ∈_ R<sup>(</sup><sup>_b×m_)</sup><sup>_×n×d_</sup> . The memory bottleneck lies in storing this intermediate tensor _E_ . For a table with one million samples and 500 features, _E_ requires approximately 250 GB of memory (with _d_ = 128 and float32 precision). TabICL addresses this by offloading _E_ to CPU memory during column-wise embedding. However, 250 GB of CPU memory remains prohibitive for many users. We therefore implement disk offloading to further alleviate memory constraints. 

**Disk offloading via memory-mapped files.** Our disk offloading implementation leverages memory-mapped files (memmap) through NumPy, which allows the operating system to handle paging between disk and memory transparently. The key components are: 1. **Pre-allocation:** Before processing begins, we pre-allocate a memory-mapped file on disk with the exact size required for the output tensor. This reserves contiguous disk space and avoids fragmentation during incremental writes. 

2. **Incremental writing:** During column-wise embedding, each batch’s output is written directly to the memory-mapped file at the corresponding indices. This streaming approach ensures that GPU memory only holds the current batch, while completed results are persisted to disk. 

3. **Periodic flushing:** To balance I/O efficiency with memory usage, we periodically flush the memory-mapped file to disk after accumulating a configurable amount of data (default is 8 GB). This prevents the operating system’s page cache from consuming excessive memory. 

4. **Automatic cleanup:** We register a weak reference finalizer for each memory-mapped file, ensuring automatic deletion when the associated tensor is garbage collected. 

**Asynchronous data transfer.** To overlap GPU computation with data movement, we employ a dedicated CUDA stream for device-to-host (D2H) transfers. The workflow operates as follows: (1) GPU tensor is asynchronously copied to a pinned CPU buffer on the copy stream; (2) a CUDA event is recorded to track completion; (3) upon event completion, data is written to the final target (CPU tensor or disk); (4) the pinned buffer is returned to a buffer pool for reuse. This pipelining hides transfer latency and improves throughput. We maintain a configurable maximum number of pending asynchronous copies (default is 4) before blocking, balancing memory usage against throughput. 

**Automatic mode selection.** We provide an inference manager that supports four offloading modes: GPU (keep everything on GPU), CPU (offload to CPU memory), DISK (offload to memory-mapped files), and AUTO (automatically choose based on available resources). In AUTO mode, the manager estimates the output tensor size and compares it against available GPU memory, CPU memory, and disk space with configurable safety factors to select the most appropriate storage backend. 

Figures H.1 and H.2 illustrate the resource utilization profiles for CPU and disk offloading respectively, processing a table with 1 million samples and 500 features (80% training, 20% test) on an H100 GPU with FlashAttention-3 and automatic mixed precision enabled. CPU offloading achieves faster execution (115s vs. 450s) but requires 250 GB of RAM. Disk offloading trades speed for accessibility, requiring only 24 GB of CPU memory and 50 GB of GPU memory while using 250 GB of disk space, which is a configuration available on most modern workstations. 

37 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
250 CPU Memory (GB) 50<br>GPU Memory (GB)<br>200 40<br>150 30<br>100 20<br>50 10<br>0<br>100<br>GPU Utilization (%)<br>50<br>0<br>0 20 40 60 80 100<br>Time (s)<br><!-- End of picture text -->

_Figure H.1._ **CPU offloading for a table with 1M samples and 500 features.** The intermediate feature embeddings tensor _E_ requires approximately 250 GB of storage. During column-wise embedding (0–35s), _E_ is progressively offloaded to CPU memory, causing CPU memory usage to increase linearly. During row-wise interaction (35–70s), batches are loaded from CPU to GPU for computation, resulting in fluctuating GPU utilization due to CPU-GPU communication overhead. The forward pass completes in 115 seconds with peak GPU memory of 50 GB. However, the 250 GB CPU memory requirement remains prohibitive for most systems. 



<!-- Start of picture text -->
CPU Memory (GB)<br>22 GPU Memory (GB) 40<br>30<br>20<br>20<br>18<br>10<br>16<br>0<br>200 Disk Usage (GB)<br>100<br>0<br>100<br>GPU Utilization (%)<br>50<br>0<br>0 100 200 300 400<br>Time (s)<br><!-- End of picture text -->

_Figure H.2._ **Disk offloading for a table with 1M samples and 500 features.** We first pre-allocate a 250 GB memory-mapped file on disk. During column-wise embedding (0–280s), feature embeddings are streamed directly to disk. The periodic drops in GPU utilization correspond to synchronization points where the asynchronous copy manager drains pending transfers and flushes data to disk. During row-wise interaction (280–350s), data is loaded from disk in batches. The total forward pass takes 450 seconds—approximately 4 _×_ slower than CPU offloading due to disk I/O latency, but with dramatically reduced memory requirements (CPU memory under 24 GB and GPU memory under 50 GB) that make million-scale inference accessible on commodity hardware. 

38 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

### **I. Quantile distribution** 

#### **I.1. Problem setup** 

For regression tasks, given an input feature vector _x_ , TabICLv2 predicts a set of conditional quantiles: 



where _αk_ represents probability levels. In this work, we set _K_ = 999 and choose uniformly spaced probability levels between _αL_ = 0 _._ 001 and _αR_ = 0 _._ 999, that is, **_α_** = _{_ 0 _._ 001 _,_ 0 _._ 002 _, . . . ,_ 0 _._ 999 _}_ . However, these raw predicted quantiles present several challenges (Chernozhukov et al., 2010; Bondell et al., 2010): 

1. **Quantile crossing** : TabICLv2 may predict non-monotonic quantiles, i.e., _Q_<sup>ˆ</sup> ( _αi_ ) _> Q_<sup>ˆ</sup> ( _αj_ ) for _αi < αj_ , which violates the fundamental property of quantile functions. 

2. **Incomplete support** : Predictions are only available for _α ∈_ [ _αL, αR_ ], leaving the extreme tails undefined. 

3. **Lack of analytical functions** : Raw quantiles do not directly provide probability density function (PDF), cumulative distribution function (CDF), or analytical moments required by many downstream applications. 

Therefore, we propose an approach to construct a probabilistic distribution from predicted quantiles, addressing these challenges by: 

1. **Monotonicity enforcement** : Correcting quantile crossing and enforcing monotonic quantiles for a valid quantile function. 

2. **Tail extrapolation** : Extending the distribution beyond [ _αL, αR_ ] using parametric exponential tail models with datainferred parameters. 

3. **Analytical statistics** : Providing closed-form expressions for PDF, CDF, and analytical moments. 

#### **I.2. Quantile function** 

The complete quantile function is defined over three regions: 



In the interior region [ _αL, αR_ ], the quantile function is modeled as a piecewise linear spline connecting the predicted quantile knots. For _α ∈_ [ _αi, αi_ +1] where _i ∈{_ 1 _,_ 2 _, . . . , K −_ 1 _}_ : 



where the slope of segment _i_ is: 



The slopes _mi_ must be non-negative for a valid quantile function, which is ensured by the monotonicity correction described later. To extrapolate the distribution beyond the observed quantile range [ _αL, αR_ ], we employ parametric exponential tail models suitable for sub-exponential distributions (e.g., Gaussian) (Beirlant et al., 2006). 

For the left tail ( _α < αL_ ): 



where _βL >_ 0 is the scale parameter and _cL_ is the intercept determined by continuity at the boundary. 

39 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

Requiring _Q_ left( _αL_ ) = _qL_ gives: 



Thus, the complete left tail formula is: 



For the right tail ( _α > αR_ ): 



where _βR >_ 0 is the scale parameter. Requiring _Q_ right( _αR_ ) = _qR_ gives: 



Thus, the complete right tail formula is: 



The derivative _dQ/dα_ is crucial for PDF computation: 



#### **I.3. Quantile crossing correction** 

A quantile crossing violation occurs when for some _i < j_ : 

ˆ ˆ _Q_ ( _αi_ ) _> Q_ ( _αj_ ) despite _αi < αj_ 

This violates the fundamental monotonicity requirement of quantile functions. We provide three ways to handle this issue: 

**(1) No correction.** The simplest approach is to ignore crossing violations. This may be acceptable when crossings are rare. However, this can lead to invalid probability densities in regions where crossings occur. 

**(2) Sorting.** A straightforward correction is to sort the predicted quantiles: 



This guarantees monotonicity with _O_ ( _K_ log _K_ ) complexity. However, sorting may destroy the correspondence between quantile values and their original probability levels, which can distort the distribution shape. 

**(3) Isotonic regression.** The optimal correction in the _L_<sup>2</sup> sense is given by isotonic regression (Barlow & Brunk, 1972), which solves: 



where _wk >_ 0 are optional weights (by default, _wk_ = 1). This problem admits a unique solution that can be computed in _O_ ( _K_ ) time via the Pool Adjacent Violators Algorithm (PAVA) (Best & Chakravarti, 1990). PAVA iteratively merges adjacent blocks that violate monotonicity and replaces each merged block with its weighted average. Despite its linear per-sample complexity, PAVA is inherently a 1D, data-dependent procedure that the sequence of merges depends on local violations and therefore introduces sequential dependencies along the quantile index. As a result, it is not straightforward 

40 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
10 1 Sort-CPU<br>Sort-GPU<br>10 0 PAVA-Scipy<br>PAVA-Numba<br>10 − 1<br>10 − 2<br>10 − 3<br>10 − 4<br>10 0 10 1 10 2 10 3 10 4 10 5 10 6<br>Batch size<br>(s)<br>Time<br>Execution<br><!-- End of picture text -->

_Figure I.1._ Execution time comparison of different monotonicity enforcement methods across batch sizes ( _K_ = 999 quantiles per sample). **Sort-GPU** (torch.sort on CUDA) achieves the best performance. **Numba** (our parallel PAVA implementation) and **Sort-CPU** show competitive scaling on CPU. **Scipy** (scipy.optimize.isotonic ~~r~~ egression) processes samples sequentially, becoming slower than Numba at large batch sizes. 

to implement PAVA as a single fully vectorized operation, which limits its practical throughput because we often need to process large batches of predictions (e.g., the entire test set). 

We leverage the just-in-time compilation of Numba (Lam et al., 2015) to optimize the PAVA implementation, achieving effective CPU parallelism. As shown in Figure I.1, the Numba-based PAVA implementation achieves competitive performance with torch.sort on CPU for large batch sizes, while being faster than the PAVA implementation of Scipy (Virtanen et al., 2020). 

However, the Numba-based implementation cannot compete with torch.sort on GPU, which relies on highly optimized parallel sorting primitives implemented in dedicated CUDA kernels. Given this performance gap and the empirical observation that quantile crossing is rare in the predictions of TabICLv2, we use sorting by default for monotonicity enforcement. 

#### **I.4. Tail parameter estimation** 

The exponential tail scale parameters _βL_ and _βR_ are estimated from the boundary quantiles using log-space linear regression. For the left tail, the model is _Q_ ( _α_ ) = _βL_ ln( _α_ ) + _cL_ . Given _K_ tail quantiles in the left tail region (default is 20), we estimate _βL_ using ordinary least squares: 



Specifically: 



For the right tail, the model is _Q_ ( _α_ ) = _−βR_ ln(1 _− α_ ) + _cR_ . The estimation is analogous: 



The estimated parameters are clamped to ensure numerical stability: _β ∈_ [ _β_ min _, β_ max] = [0 _._ 01 _,_ 100]. 

#### **I.5. Cumulative distribution function (CDF)** 

The CDF _F_ ( _z_ ) = _P_ ( _Z ≤ z_ ) is the inverse of the quantile function: _F_ ( _z_ ) = _Q_<sup>_−_1</sup> ( _z_ ). 

**Spline region** ( _z ∈_ [ _qi, qi_ +1)): 



41 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

**Left tail** ( _z < qL_ ): From _z_ = _qL_ + _βL_ ln( _α/αL_ ), we solve for _α_ : 



**Right tail** ( _z > qR_ ): 



#### **I.6. Probability density function (PDF)** 

The PDF is related to the quantile function derivative by the inverse function theorem: 



The PDF computation procedure is: 

1. Compute _α_ = _F_ ( _z_ ) using the appropriate CDF formula 

2. Compute<sup>_d_</sup> _dα_<sup>_<u>Q</u>_at</sup><sup>_α_using the appropriate derivative formula</sup> 



**Spline region** ( _z ∈_ [ _qi, qi_ +1)): 

**Left tail** ( _z < qL_ ): 

**Right tail** ( _z > qR_ ): 



The log probability density is computed directly to avoid numerical issues with very small densities: 



#### **I.7. Continuous ranked probability score (CRPS)** 

The CRPS for a distribution with CDF _F_ and observation _z_ is: 



where **1** _y≥z_ is the indicator function. This can be equivalently expressed in quantile space: 



where _ρα_ ( _u_ ) = _u_ ( _α −_ **1** _u<_ 0) is the pinball loss. The CRPS decomposes as: 



We compute CRPS analytically by integrating over each region. 

42 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### I.7.1. CRPS CONTRIBUTION FROM SPLINE REGION 

For segment _i_ with _α ∈_ [ _αi, αi_ +1] and _Q_ ( _α_ ) = _qi_ + _mi_ ( _α − αi_ ), we let _r_ = min(max( _F_ ( _z_ ) _, αi_ ) _, αi_ +1) be the clamped CDF value. The contribution to CRPS from segment _i_ is: 

where: 



For the first integral ( _α ≤ F_ ( _z_ )): 



Computing each integral: 



Substituting yields the formula for _I_ 1<sup>(</sup><sup>_i_).The second integral</sup><sup>_I_</sup> 2<sup>(</sup><sup>_i_)</sup> is computed analogously. The total spline CRPS is: 



#### I.7.2. CRPS CONTRIBUTION FROM EXPONENTIAL LEFT TAIL 

For the left exponential tail with _Q_ ( _α_ ) = _qL_ + _βL_ ln( _α/αL_ ), let _α_ ˜ = min( _F_ ( _z_ ) _, αL_ ) be the clamped CDF value and _bL_ = _qL − βL_ ln _αL_ . We have: 



where: 



#### I.7.3. CRPS CONTRIBUTION FROM EXPONENTIAL RIGHT TAIL 

˜ For the right exponential tail with _Q_ ( _α_ ) = _qR − βR_ ln((1 _− α_ ) _/_ (1 _− αR_ )), let _α_ = max( _F_ ( _z_ ) _, αR_ ) be the clamped CDF value. We have: 



where _aR_ = _−βR_ , _bR_ = _qR_ + _βR_ ln(1 _− αR_ ), and: 





**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **I.8. Moment calculations** 

#### I.8.1. MEAN 

The mean of a distribution can be computed as: 



**Spline contribution** : 



**Left tail contribution:** 



**Right tail contribution:** 



**Total mean:** 



#### I.8.2. VARIANCE 

The variance is computed as: 

where: 



**Spline contribution** : For a linear segment: 



Thus: 



**Left tail contribution:** 



**Right tail contribution:** 



44 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **I.9. Empirical validation on synthetic regression tasks** 

To validate that QuantileDistribution correctly constructs probability distributions from the predicted quantiles of TabICLv2, we design four synthetic regression datasets with known ground-truth distributions. This allows direct comparison between predicted and true distributional quantities, PDF, and CDF. 

#### I.9.1. SYNTHETIC REGRESSION DATASETS 

**Dataset 1: Quadratic with homoscedastic gaussian noise.** This serves as a simple baseline with constant variance: 



The true conditional distribution is _p_ ( _y|x_ ) = _N_ (0 _._ 15 _x_<sup>2</sup> _−_ 0 _._ 5 _,_ 0 _._ 25<sup>2</sup> ). The predictive distribution should exhibit uniform spread across all _x_ values, with symmetric Gaussian PDFs centered on the quadratic mean function. 

**Dataset 2: Sinusoidal with heteroscedastic noise.** This dataset tests the ability of TabICLv2 to capture input-dependent uncertainty: 



The true conditional distribution is _p_ ( _y|x_ ) = _N_ (sin(2 _x_ ) + 0 _._ 2 _x, σ_ ( _x_ )<sup>2</sup> ). The noise variance increases with _|x|_ , requiring the model to predict wider quantile intervals at the boundaries than at the center. 

**Dataset 3: Step function with noise.** This dataset tests behavior at discontinuities: 



The true conditional distribution is _p_ ( _y|x_ ) = _N_ (sign( _x_ ) _,_ 0 _._ 3<sup>2</sup> ). At _x_ = 0, the model must handle the abrupt transition between two distinct modes. 

**Dataset 4: Linear with heavy-tailed noise.** This dataset introduces heavy-tailed behavior via a Gaussian mixture to test tail extrapolation: 



where _w_ = 0 _._ 1 (10% outlier weight), _σ_ 1 = 0 _._ 2 (inlier scale), and _σ_ 2 = 0 _._ 8 (outlier scale). The true conditional distribution is a Gaussian mixture _p_ ( _y|x_ ) = 0 _._ 9 _· N_ (0 _._ 3 _x,_ 0 _._ 2<sup>2</sup> ) + 0 _._ 1 _· N_ (0 _._ 3 _x,_ 0 _._ 8<sup>2</sup> ). This creates heavier tails than a pure Gaussian, testing whether the exponential tail model adequately captures extreme quantiles. 

#### I.9.2. VISUALIZATION AND ANALYSIS 

Figure I.2 presents a comprehensive 6-row _×_ 4-column visualization comparing predicted distributions (solid lines) against ground-truth distributions (dashed lines). Each column corresponds to one dataset: 

1. **Row 1 (quantile lines)** : Training data overlaid with predicted quantile curves. The median tracks the distribution center, while extreme quantiles delineate tail behavior. For the heteroscedastic dataset (column 2), the quantile band visibly widens toward the boundaries _|x|_ = 3, correctly capturing input-dependent variance. 

2. **Row 2 (quantile functions)** : The quantile function for three representative inputs _x ∈{−_ 2 _,_ 0 _,_ 2 _}_ . Predicted curves (solid) closely match the true quantile functions (dashed) across all datasets. The crossing correction is evident, and the exponential tail extrapolation smoothly extends the curves beyond [ _αL, αR_ ] = [0 _._ 001 _,_ 0 _._ 999]. 

3. **Row 3 (PDF)** : Predicted PDFs (solid) align well with true PDFs (dashed), capturing both the location and spread of the conditional distributions. The heteroscedastic dataset shows narrower peaks at _x_ = 0 and broader peaks at _x_ = _±_ 2, matching the ground truth. 

4. **Row 4 (CDF)** : The smooth S-curves of predicted CDFs (solid) closely follow the true CDFs (dashed). 

45 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

5. **Row 5 (Density heatmaps)** : A 2D visualization of the conditional density _f_ ( _y|x_ ) across the input domain (log scale). The heteroscedastic dataset shows a funnel-shaped density widening with _|x|_ , while the step function exhibits two distinct horizontal bands separated at _x_ = 0. 

6. **Row 6 (resampled data)** : Synthetic samples drawn from the learned distribution via inverse transform sampling ( _y_ = _Q_ ( _U_ ) where _U ∼_ Uniform(0 _,_ 1)). The resampled points (blue) closely match the original training data (gray). 

46 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
Quadratic + Gaussian Sinusoidal + Heteroscedastic Step Function Linear + Heavy-tailed<br>4 4 4 4<br>Training data α  = 0 . 0001 α  = 0 . 001 α  = 0 . 5 α  = 0 . 999 α  = 0 . 9999<br>2 2 2 2<br>0 0 0 0<br>− 2 − 2 − 2 − 2<br>− 4 − 4 − 4 − 4<br>− 2 0 2 − 2 0 2 − 2 0 2 − 2 0 2<br>x x x x<br>4 4 4 4<br>x=-2 (true) x=0 (true) x=2 (true)<br>2 2 x=-2 (pred) x= 0 (pred)2 x=2 (pred) 2<br>0 0 0 0<br>− 2 − 2 − 2 − 2<br>− 4 − 4 − 4 − 4<br>0 . 00 0 . 25 0 . 50 0 . 75 1 . 00 0 . 00 0 . 25 0 . 50 0 . 75 1 . 00 0 . 00 0 . 25 0 . 50 0 . 75 1 . 00 0 . 00 0 . 25 0 . 50 0 . 75 1 . 00<br>α α α α<br>2 . 0<br>2 . 0 x=-2 (true) x=0 (true) x=2 (true) 4<br>1 . 5 3 x=-2 (pred) x= 0 (pred)1 . 5 x=2 (pred) 3<br>1 . 0 2 1 . 0 2<br>0 . 5 1 0 . 5 1<br>0 . 0 0 0 . 0 0<br>− 4 − 2 0 2 4 − 4 − 2 0 2 4 − 4 − 2 0 2 4 − 4 − 2 0 2 4<br>y y y y<br>1 . 0 1 . 0 1 . 0 1 . 0<br>x=-2 (true) x=0 (true) x=2 (true)<br>0 . 8 0 . 8 x=-2 (pred) x= 0 (pred)0 . 8 x=2 (pred) 0 . 8<br>0 . 6 0 . 6 0 . 6 0 . 6<br>0 . 4 0 . 4 0 . 4 0 . 4<br>0 . 2 0 . 2 0 . 2 0 . 2<br>0 . 0 0 . 0 0 . 0 0 . 0<br>− 4 − 2 0 2 4 − 4 − 2 0 2 4 − 4 − 2 0 2 4 − 4 − 2 0 2 4<br>y y y y<br>Density Map Density Map Density Map Density Map<br>4 4 4 4<br>10 − 2 10 − 1 10 0 10 − 2 10 − 1 10 0 10 − 2 10 − 1 10 0 10 − 2 10 − 1 10 0<br>2 Density 2 Density 2 Density 2 Density<br>0 0 0 0<br>− 2 − 2 − 2 − 2<br>− 4 − 4 − 4 − 4<br>− 2 0 2 − 2 0 2 − 2 0 2 − 2 0 2<br>x x x x<br>Resampled Data Resampled Data Resampled Data Resampled Data<br>4 4 4 4<br>Original Original Original Original<br>Resampled Resampled Resampled Resampled<br>2 2 2 2<br>0 0 0 0<br>− 2 − 2 − 2 − 2<br>− 4 − 4 − 4 − 4<br>− 2 0 2 − 2 0 2 − 2 0 2 − 2 0 2<br>x x x x<br>y y y y<br>)( Qα )( Qα )( Qα )( Qα<br>PDF PDF PDF PDF<br>CDF CDF CDF CDF<br>y y y y<br>y y y y<br><!-- End of picture text -->

_Figure I.2._ **Validation of QuantileDistribution on four synthetic regression tasks with known ground-truth distributions.** 

47 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

### **J. Detailed results on the TabArena benchmark** 

#### **J.1. Aggregation metrics** 

TabArena (Erickson et al., 2025) evaluates models using task-specific error metrics and aggregates results across datasets using several complementary metrics. We briefly summarize these metrics below. 

**Per-dataset error metrics.** For each dataset _i_ , TabArena computes an error metric err _i_ by averaging over all outer cross-validation folds: 

- **Binary classification** : 1 _−_ ROC AUC 

- **Multiclass classification** : Log-Loss 

- **Regression** : RMSE 

**Elo rating.** Elo is a pairwise comparison-based rating system where each model’s rating predicts its expected win probability against others. A 400-point Elo gap corresponds to a 10:1 (91%) expected win rate. For two models _A_ and _B_ with ratings _RA_ and _RB_ , the expected win probability of _A_ is: 



Elo is based solely on wins, ties, or losses and ignores the magnitude of performance differences. This ensures each dataset contributes equally to the final ranking, avoiding bias toward certain domains or dataset properties. TabArena calibrates 1000 Elo to the performance of default RandomForest and uses 200 rounds of bootstrapping for 95% confidence intervals. 

**Improvability.** Improvability measures the relative error gap between a method and the best-performing method on each dataset, then averages across datasets. For a model _m_ on dataset _i_ : 



where err<sup>_∗_</sup> _i_<sup>= min</sup><sup>_m′_err</sup><sup>_i_(</sup><sup>_m′_) is the error of the best method on dataset</sup><sup>_i_.Improvability is always between 0% (optimal)</sup> and 100%. Unlike Elo, improvability is sensitive to the magnitude of performance differences, making it more informative for practitioners who care about how much a method lags behind the best. 

**Average rank.** For each dataset, models are ranked by their error (rank 1 is best). The average rank is simply the mean rank across all datasets: 



Lower average rank indicates better overall performance. 

**Discussion.** Each aggregation metric has its own strengths and limitations. Elo treats all datasets equally regardless of performance gaps, improvability captures the magnitude of differences, and rank-based metrics are robust to outliers. We primarily report improvability in the main paper for its interpretability, but provide Elo and rankings in this appendix for completeness. Across all metrics, TabICLv2 consistently achieves state-of-the-art performance. 

#### **J.2. Results on all datasets** 

Figures in the section present the complete results on all 51 TabArena datasets. 

**Ranking and Elo.** TabICLv2 (default) achieves an average rank of 4.82, outperforming AutoGluon 1.4 (extreme, 4h) at 5.24 and RealTabPFN-2.5 (T+E) at 5.88. Here, AutoGluon (extreme, 4h) refers to AutoGluon (Erickson et al., 2020) with the best ~~q~~ uality preset and a 4-hour training budget, which ensembles multiple model families with extensive hyperparameter tuning. Despite a single forward-pass without any tuning, TabICLv2 achieves better performance with this heavily optimized ensemble system. 

48 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

**Pairwise win rates.** While TabICLv2 ranks below AutoGluon 1.5 (extreme, 4h) in average rank (4.82 vs. 3.88), the pairwise win rate matrix (Figure J.2) reveals a more nuanced picture: TabICLv2 achieves a 57% win rate against AutoGluon 1.5 and a 59% win rate against RealTabPFN-2.5 (T+E). This indicates that TabICLv2 wins on the majority of datasets in head-to-head comparisons. 

The discrepancy between win rate and average rank arises from how these metrics handle the magnitude of performance differences. Win rate only counts wins and losses, treating all victories equally. In contrast, average rank penalizes methods that perform poorly on certain datasets, even if they win on most others. This suggests that while TabICLv2 wins more often, AutoGluon 1.5 may achieve more consistent rankings across datasets, avoiding the occasional poor performance that can inflate average rank. This suggests that TabICLv2 may struggle on specific datasets that fall outside its pretraining distribution. 

**Pareto efficiency.** As shown in Figures J.3 and J.4, TabICLv2 dominates the Pareto front of both improvability vs. runtime and Elo vs. runtime among all tabular foundation models. TabICLv2 achieves the best trade-off between predictive performance and computational cost, being both faster and more accurate than competing TFMs including RealTabPFN-2.5, TabPFN-2.5, TabICL, LimiX, and Mitra. 



<!-- Start of picture text -->
20 . 0 17 . 5 15 . 0 12 . 5 10 . 0 7 . 5 5 . 0<br>KNN (T+E) [21.57] [3.88] AutoGluon 1.5 (extreme, 4h)<br>Linear (T+E) [21.49] [4.82] TabICLv2 (default)<br>RandomForest (T+E) [18.14] [5.24] AutoGluon 1.4 (extreme, 4h)<br>FastaiMLP (T+E) [17.51] [5.88] RealTabPFN-2.5 (T+E)<br>ExtraTrees (T+E) [17.29] [9.04] RealMLP (T+E)<br>EBM (T+E) [15.94] [9.20] AutoGluon 1.4 (best, 4h)<br>TorchMLP (T+E) [15.80] [9.82] TabDPT (T+E)<br>xRFM (T+E) [14.18] [10.24] LimiX (default)<br>XGBoost (T+E) [13.49] [11.27] TabM (T+E)<br>TabICL (default) [13.20] [11.84] LightGBM (T+E)<br>ModernNCA (T+E) [12.84] [12.27] CatBoost (T+E)<br>Mitra (default) [12.74] [12.30] TabPFNv2 (T+E)<br><!-- End of picture text -->

_Figure J.1._ **Critical difference diagram on the TabArena benchmark.** 

49 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
AutoGluon 1.5 (extreme, 4h) 43 73 61 88 90 78 82 88 98 96 78 78 92 90 96 96 98 96 98 96 98 96100<br>TabICLv2 (default) 57 61 59 76 75 84 78 86 82 82 82 88 88 94 82 88 88 90 94 92 9210096<br>AutoGluon 1.4 (extreme, 4h) 27 39 45 78 78 69 69 92 96 98 71 78 86 80 98 94 96 94 94 98 98 96100<br>RealTabPFN-2.5 (T+E) 39 41 55 75 75 67 75 76 82 84 86 80 76 78 84 88 90 88 94 92 94 92 98<br>RealMLP (T+E) 12 24 22 25 45 51 45 63 67 75 61 65 71 67 78 88 82 86 92 92 94 94 98<br>AutoGluon 1.4 (best, 4h) 10 25 22 25 55 47 53 65 75 76 53 61 63 65 84 78 90 82 82 88 90 94 96<br>TabDPT (T+E) 22 16 31 33 49 53 57 65 65 65 57 65 61 59 65 73 78 73 86 82 84 86 94<br>LimiX (default) 18 22 31 25 55 47 43 63 67 65 61 61 65 59 69 69 76 75 73 78 78 92 84<br>TabM (T+E) 12 14 8 24 37 35 35 37 59 65 53 53 61 61 67 63 82 76 80 88 78 92 92<br>LightGBM (T+E) 2 18 4 18 33 25 35 33 41 59 49 51 53 61 76 63 75 76 82 80 88 96 96<br>CatBoost (T+E) 4 18 2 16 25 24 35 35 35 41 47 53 55 59 69 63 80 73 82 76 90 94 96<br>TabPFNv2 (T+E) 22 18 29 14 39 47 43 39 47 51 53 54 57 52 51 59 59 69 65 71 65 86 84<br>Mitra (default) 22 12 22 20 35 39 35 39 47 49 47 46 53 57 53 51 61 73 63 73 59 88 82<br>ModernNCA (T+E) 8 12 14 24 29 37 39 35 39 47 45 43 47 55 55 55 65 67 76 75 78 84 86<br>TabICL (default) 10 6 20 22 33 35 41 41 39 39 41 48 43 45 49 55 63 65 69 71 63 92 92<br>XGBoost (T+E) 4 18 2 16 22 16 35 31 33 24 31 49 47 45 51 55 75 75 75 73 88 94 94<br>xRFM (T+E) 4 12 6 12 12 22 27 31 37 37 37 41 49 45 45 45 55 57 73 71 80 92 92<br>TorchMLP (T+E) 2 12 4 10 18 10 22 24 18 25 20 41 39 35 37 25 45 49 69 73 71 84 88<br>EBM (T+E) 4 10 6 12 14 18 27 25 24 24 27 31 27 33 35 25 43 51 63 61 65 94 86<br>ExtraTrees (T+E) 2 6 6 6 8 18 14 27 20 18 18 35 37 24 31 25 27 31 37 49 65 76 90<br>FastaiMLP (T+E) 4 8 2 8 8 12 18 22 12 20 24 29 27 25 29 27 29 27 39 51 59 88 80<br>RandomForest (T+E) 2 8 2 6 6 10 16 22 22 12 10 35 41 22 37 12 20 29 35 35 41 76 88<br>Linear (T+E) 4 0 4 8 6 6 14 8 8 4 6 14 12 16 8 6 8 16 6 24 12 24 41<br>KNN (T+E) 0 4 0 2 2 4 6 16 8 4 4 16 18 14 8 6 8 12 14 10 20 12 59<br><!-- End of picture text -->



<!-- Start of picture text -->
Win Rate (%)<br>100<br>80<br>60<br>40<br>20<br><!-- End of picture text -->

_Figure J.2._ **Winrate matrix on the TabArena benchmark.** 



<!-- Start of picture text -->
Linear Linear<br>Default KNN Default KNN<br>Tuned RandomForest Tuned RandomForest<br>40% Tuned + Ens. ExtraTrees 40% Tuned + Ens. ExtraTrees<br>Pareto Front FastaiMLP Pareto Front FastaiMLP<br>EBM EBM<br>TorchMLP TorchMLP<br>30% Mitra 30% Mitra<br>TabICL TabICL<br>LimiX LimiX<br>TabPFNv2 TabPFNv2<br>20% xRFM 20% xRFM<br>XGBoost XGBoost<br>ModernNCA ModernNCA<br>LightGBM LightGBM<br>10% CatBoost 10% CatBoost<br>TabM TabM<br>TabDPT TabDPT<br>RealMLP RealMLP<br>10 − 1 10 0 10 1 10 2 10 3 10 4 RealTabPFN-2.5 10 − 1 10 0 10 1 10 2 RealTabPFN-2.5<br>TabICLv2 TabICLv2<br>Train time per 1K samples (s) (median) Inference time per 1K samples (s) (median)<br>(a)  Pareto front of improvability and train time (b)  Pareto front of improvability and inference time<br>Optimal Optimal<br>Improvability (%) Improvability (%)<br><!-- End of picture text -->

_Figure J.3._ **Pareto front of improvability and train/inference time on the TabArena benchmark.** 

50 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
TabICLv2 TabICLv2<br>1800 RealTabPFN-2.5 1800 RealTabPFN-2.5<br>RealMLP RealMLP<br>1600 TabDPT 1600 TabDPT<br>TabM TabM<br>LightGBM LightGBM<br>1400 LimiX 1400 LimiX<br>CatBoost CatBoost<br>ModernNCA ModernNCA<br>1200 XGBoost 1200 XGBoost<br>TabPFNv2 TabPFNv2<br>Mitra Mitra<br>1000 1000<br>TabICL TabICL<br>xRFM xRFM<br>800 Default EBM 800 Default EBM<br>Tuned TorchMLP Tuned TorchMLP<br>Tuned + Ens. FastaiMLP Tuned + Ens. FastaiMLP<br>600 Pareto Front ExtraTrees 600 Pareto Front ExtraTrees<br>RandomForest RandomForest<br>10 − 1 10 0 10 1 10 2 10 3 10 4 KNN 10 − 1 10 0 10 1 10 2 KNN<br>Linear Linear<br>Train time per 1K samples (s) (median) Inference time per 1K samples (s) (median)<br>(a)  Pareto front of Elo and train time (b)  Pareto front of Elo and inference time<br>Optimal Optimal<br>Elo Elo<br><!-- End of picture text -->

_Figure J.4._ **Pareto front of Elo and train/inference time on the TabArena benchmark.** 



<!-- Start of picture text -->
Partially imputed Tuned<br>Default Tuned + Ensembled<br>TabICLv2<br>RealTabPFN-2.5<br>RealMLP<br>TabDPT<br>TabM<br>LightGBM<br>LimiX<br>CatBoost<br>ModernNCA<br>XGBoost<br>TabPFNv2<br>Mitra<br>TabICL<br>xRFM<br>EBM<br>TorchMLP<br>FastaiMLP<br>ExtraTrees AutoG luon 1.4 (best, 4h)<br>RandomForest A utoGluon 1.4 (extreme, 4h)<br>K NN<br>AutoGluon 1.5 (extreme, 4h)<br>Lin ear<br>800 1000 1200 1400 1600 1800<br>Elo<br><!-- End of picture text -->

_Figure J.5._ **TabArena Elo.** 

51 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **J.3. Results on binary classification datasets** 

Figures in this section present results on 24 binary classification datasets in TabArena. TabICLv2 achieves an average rank of 4.43, placing second behind AutoGluon 1.5 (extreme, 4h) at 3.83. Notably, TabICLv2 substantially outperforms RealTabPFN-2.5 (T+E) at 6.90. On the Pareto fronts, TabICLv2 remains the strongest among all tabular foundation models for binary classification. 



<!-- Start of picture text -->
20 . 0 17 . 5 15 . 0 12 . 5 10 . 0 7 . 5 5 . 0<br>KNN (T+E) [21.50] [3.83] AutoGluon 1.5 (extreme, 4h)<br>Linear (T+E) [20.33] [4.43] TabICLv2 (default)<br>RandomForest (T+E) [19.23] [5.00] AutoGluon 1.4 (extreme, 4h)<br>ExtraTrees (T+E) [18.43] [6.90] RealTabPFN-2.5 (T+E)<br>FastaiMLP (T+E) [16.47] [9.37] AutoGluon 1.4 (best, 4h)<br>TorchMLP (T+E) [15.50] [9.70] TabICL (default)<br>EBM (T+E) [15.23] [9.80] RealMLP (T+E)<br>xRFM (T+E) [15.17] [10.35] LimiX (default)<br>ModernNCA (T+E) [13.87] [10.83] TabM (T+E)<br>XGBoost (T+E) [13.47] [11.13] TabDPT (T+E)<br>TabPFNv2 (T+E) [12.98] [11.60] LightGBM (T+E)<br>CatBoost (T+E) [12.73] [12.13] Mitra (default)<br><!-- End of picture text -->

_Figure J.6._ **Critical difference diagram on binary classification datasets of the TabArena benchmark.** 



<!-- Start of picture text -->
Linear Linear<br>Default KNN Default KNN<br>Tuned RandomForest Tuned RandomForest<br>40% Tuned + Ens. ExtraTrees 40% Tuned + Ens. ExtraTrees<br>Pareto Front EBM Pareto Front EBM<br>FastaiMLP FastaiMLP<br>TabPFNv2 TabPFNv2<br>30% Mitra 30% Mitra<br>xRFM xRFM<br>TorchMLP TorchMLP<br>XGBoost XGBoost<br>20% ModernNCA 20% ModernNCA<br>LimiX LimiX<br>LightGBM LightGBM<br>CatBoost CatBoost<br>10% TabM 10% TabM<br>RealMLP RealMLP<br>TabDPT TabDPT<br>TabICL TabICL<br>10 − 1 10 0 10 1 10 2 10 3 10 4 RealTabPFN-2.5 10 − 2 10 − 1 10 0 10 1 10 2 10 3 RealTabPFN-2.5<br>TabICLv2 TabICLv2<br>Train time per 1K samples (s) (median) Inference time per 1K samples (s) (median)<br>(a)  Pareto front of improvability and train time (b)  Pareto front of improvability and inference time<br>Optimal Optimal<br>Improvability (%) Improvability (%)<br><!-- End of picture text -->

_Figure J.7._ **Pareto front of improvability and train/inference time on binary classification datasets of the TabArena benchmark.** 

52 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
TabICLv2 TabICLv2<br>1800 RealTabPFN-2.5 1800 RealTabPFN-2.5<br>RealMLP RealMLP<br>TabICL TabICL<br>1600 TabM 1600 TabM<br>LightGBM LightGBM<br>1400 LimiX 1400 LimiX<br>TabDPT TabDPT<br>CatBoost CatBoost<br>1200 XGBoost 1200 XGBoost<br>Mitra Mitra<br>1000 TabPFNv2 1000 TabPFNv2<br>ModernNCA ModernNCA<br>EBM EBM<br>800 800<br>Default xRFM Default xRFM<br>Tuned TorchMLP Tuned TorchMLP<br>600 Tuned + Ens. FastaiMLP 600 Tuned + Ens. FastaiMLP<br>Pareto Front ExtraTrees Pareto Front ExtraTrees<br>RandomForest RandomForest<br>400 400<br>10 − 1 10 0 10 1 10 2 10 3 10 4 Linear 10 − 2 10 − 1 10 0 10 1 10 2 10 3 Linear<br>KNN KNN<br>Train time per 1K samples (s) (median) Inference time per 1K samples (s) (median)<br>(a)  Pareto front of Elo and train time (b)  Pareto front of Elo and inference time<br>Optimal Optimal<br>Elo Elo<br><!-- End of picture text -->

_Figure J.8._ **Pareto front of Elo and train/inference time on binary classification datasets of the TabArena benchmark.** 

#### **J.4. Results on multiclass classification datasets** 

Figures in the section present results on 14 multiclass classification datasets in TabArena. On multiclass classification, TabICLv2 (default) achieves an average rank of 6.75, behind AutoGluon 1.5 (extreme, 4h) at 4.00, RealTabPFN-2.5 (T+E) at 4.25, and AutoGluon 1.4 (extreme, 4h) at 4.50. While TabICLv2 does not surpass RealTabPFN-2.5 (T+E) on this subset, it is important to note that RealTabPFN-2.5 employs tuning and ensembling whereas TabICLv2 does not perform any tuning. In addition, TabICLv2 substantially outperforms other TFMs, such as TabPFNv2 (T+E) at 8.62 and LimiX (default) at 9.31. 



<!-- Start of picture text -->
22 . 5 20 . 0 17 . 5 15 . 0 12 . 5 10 . 0 7 . 5 5 . 0<br>Linear (T+E) [22.12] [4.00] AutoGluon 1.5 (extreme, 4h)<br>KNN (T+E) [21.50] [4.25] RealTabPFN-2.5 (T+E)<br>FastaiMLP (T+E) [16.88] [4.50] AutoGluon 1.4 (extreme, 4h)<br>TorchMLP (T+E) [16.75] [6.75] TabICLv2 (default)<br>ExtraTrees (T+E) [15.25] [8.62] TabPFNv2 (T+E)<br>RandomForest (T+E) [15.25] [9.31] LimiX (default)<br>EBM (T+E) [15.12] [9.88] RealMLP (T+E)<br>TabICL (default) [14.81] [11.00] AutoGluon 1.4 (best, 4h)<br>xRFM (T+E) [14.50] [11.25] Mitra (default)<br>ModernNCA (T+E) [14.38] [11.62] TabM (T+E)<br>XGBoost (T+E) [14.00] [12.25] TabDPT (T+E)<br>LightGBM (T+E) [13.38] [12.62] CatBoost (T+E)<br><!-- End of picture text -->

_Figure J.9._ **Critical difference diagram on multiclass classification datasets of the TabArena benchmark.** 

53 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
KNN KNN<br>70% Default Tuned LinearFastaiMLP 70% Default Tuned LinearFastaiMLP<br>Tuned + Ens. TabICL Tuned + Ens. TabICL<br>60% Pareto Front ExtraTrees 60% Pareto Front ExtraTrees<br>TorchMLP TorchMLP<br>50% RandomForest 50% RandomForest<br>EBM EBM<br>40% TabDPT 40% TabDPT<br>xRFM xRFM<br>Mitra Mitra<br>30% ModernNCA 30% ModernNCA<br>XGBoost XGBoost<br>20% TabM 20% TabM<br>LightGBM LightGBM<br>10% TabPFNv2 10% TabPFNv2<br>CatBoost CatBoost<br>RealMLP RealMLP<br>0% TabICLv2 0% TabICLv2<br>10 0 10 1 10 2 10 3 10 4 10 5 LimiX 10 − 1 10 0 10 1 10 2 LimiX<br>RealTabPFN-2.5 RealTabPFN-2.5<br>Train time per 1K samples (s) (median) Inference time per 1K samples (s) (median)<br>(a)  Pareto front of improvability and train time (b)  Pareto front of improvability and inference time<br>Optimal Optimal<br>Improvability (%) Improvability (%)<br><!-- End of picture text -->

_Figure J.10._ **Pareto front of improvability and train/inference time on multiclass classification datasets of the TabArena benchmark.** 



<!-- Start of picture text -->
2000 RealTabPFN-2.5 2000 RealTabPFN-2.5<br>TabICLv2 TabICLv2<br>LimiX LimiX<br>TabPFNv2 TabPFNv2<br>1500 RealMLP 1500 RealMLP<br>TabM TabM<br>Mitra Mitra<br>CatBoost CatBoost<br>1000 LightGBM 1000 LightGBM<br>TabDPT TabDPT<br>XGBoost XGBoost<br>ModernNCA ModernNCA<br>500 xRFM 500 xRFM<br>EBM EBM<br>Default TabICL Default TabICL<br>Tuned RandomForest Tuned RandomForest<br>0 Tuned + Ens. ExtraTrees 0 Tuned + Ens. ExtraTrees<br>Pareto Front TorchMLP Pareto Front TorchMLP<br>FastaiMLP FastaiMLP<br>10 0 10 1 10 2 10 3 10 4 10 5 Linear 10 − 1 10 0 10 1 10 2 Linear<br>KNN KNN<br>Train time per 1K samples (s) (median) Inference time per 1K samples (s) (median)<br>(a)  Pareto front of Elo and train time (b)  Pareto front of Elo and inference time<br>Optimal Optimal<br>Elo Elo<br><!-- End of picture text -->

_Figure J.11._ **Pareto front of Elo and train/inference time on multiclass classification datasets of the TabArena benchmark.** 

54 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **J.5. Results on regression datasets** 

Figures in this section present results on 13 regression datasets in TabArena. On regression tasks, TabICLv2 (default) achieves an average rank of 4.54, tying with RealTabPFN-2.5 (T+E) and trailing only AutoGluon 1.5 (extreme, 4h) at 3.92. Interestingly, TabDPT (T+E) achieves a competitive rank of 5.31 on regression, substantially better than its performance on classification tasks where it ranks much lower. TabDPT is pretrained on real-world datasets rather than synthetic data. However, its strong regression performance raises questions about potential data leakage between the training corpus of TabDPT and the regression datasets of TabArena. 



<!-- Start of picture text -->
22 . 5 20 . 0 17 . 5 15 . 0 12 . 5 10 . 0 7 . 5 5 . 0<br>Linear (T+E) [23.77] [3.92] AutoGluon 1.5 (extreme, 4h)<br>KNN (T+E) [21.77] [4.54] TabICLv2 (default)<br>FastaiMLP (T+E) [20.31] [4.54] RealTabPFN-2.5 (T+E)<br>TabICL (default) [20.27] [5.31] TabDPT (T+E)<br>EBM (T+E) [18.08] [6.23] AutoGluon 1.4 (extreme, 4h)<br>RandomForest (T+E) [17.38] [6.77] RealMLP (T+E)<br>TorchMLP (T+E) [15.92] [7.69] AutoGluon 1.4 (best, 4h)<br>ExtraTrees (T+E) [15.92] [9.54] ModernNCA (T+E)<br>Mitra (default) [15.04] [10.54] LimiX (default)<br>XGBoost (T+E) [13.23] [11.00] CatBoost (T+E)<br>TabPFNv2 (T+E) [13.00] [11.46] LightGBM (T+E)<br>TabM (T+E) [12.08] [11.69] xRFM (T+E)<br>Figure J.12. Critical difference diagram on regression datasets of the TabArena benchmark.<br>Linear Linear<br>40% KNN 40% Default KNN<br>TabICL Tuned TabICL<br>FastaiMLP Tuned + Ens. FastaiMLP<br>LimiX Pareto Front LimiX<br>30% EBM 30% EBM<br>RandomForest RandomForest<br>ExtraTrees ExtraTrees<br>Default<br>TorchMLP TorchMLP<br>Tuned<br>20% Tuned + Ens. Mitra 20% Mitra<br>XGBoost XGBoost<br>Pareto Front<br>TabPFNv2 TabPFNv2<br>LightGBM LightGBM<br>10% CatBoost 10% CatBoost<br>xRFM xRFM<br>ModernNCA ModernNCA<br>TabM TabM<br>0% RealMLP 0% RealMLP<br>TabDPT TabDPT<br>10 − 1 10 0 10 1 10 2 10 3 10 4 TabICLv2 10 − 2 10 − 1 10 0 10 1 10 2 TabICLv2<br>RealTabPFN-2.5 RealTabPFN-2.5<br>Train time per 1K samples (s) (median) Inference time per 1K samples (s) (median)<br>(a)  Pareto front of improvability and train time (b)  Pareto front of improvability and inference time<br>Optimal Optimal<br>Improvability (%) Improvability (%)<br><!-- End of picture text -->

_Figure J.13._ **Pareto front of improvability and train/inference time on regression datasets of the TabArena benchmark.** 

55 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
RealTabPFN-2.5 RealTabPFN-2.5<br>2000 2000<br>TabDPT TabDPT<br>TabICLv2 TabICLv2<br>1750 RealMLP 1750 RealMLP<br>ModernNCA ModernNCA<br>1500 CatBoost 1500 CatBoost<br>LightGBM LightGBM<br>1250 xRFM 1250 xRFM<br>TabM TabM<br>1000 LimiX 1000 LimiX<br>XGBoost XGBoost<br>TabPFNv2 TabPFNv2<br>750 750<br>Mitra Mitra<br>TorchMLP TorchMLP<br>500 Default ExtraTrees 500 Default ExtraTrees<br>Tuned RandomForest Tuned RandomForest<br>250 Tuned + Ens. EBM 250 Tuned + Ens. EBM<br>Pareto Front FastaiMLP Pareto Front FastaiMLP<br>0 TabICL 0 TabICL<br>10 − 1 10 0 10 1 10 2 10 3 10 4 KNN 10 − 2 10 − 1 10 0 10 1 10 2 KNN<br>Linear Linear<br>Train time per 1K samples (s) (median) Inference time per 1K samples (s) (median)<br>(a)  Pareto front of Elo and train time (b)  Pareto front of Elo and inference time<br>Optimal Optimal<br>Elo Elo<br><!-- End of picture text -->

_Figure J.14._ **Pareto front of Elo and train/inference time on regression datasets of the TabArena benchmark.** 

### **K. Detailed results on the TALENT benchmark** 

#### **K.1. Benchmark overview** 

TALENT (Ye et al., 2024) comprises 300 datasets spanning three task types: 

- **Binary classification** : 120 datasets 

- **Multiclass classification** : 80 datasets 

- **Regression** : 100 datasets 

Each dataset is split into 64%/16%/20% for training, validation, and test sets, respectively. Hyperparameters are selected on the validation set based on accuracy, and final performance is reported on the held-out test set. 

**Evaluation metrics.** Following the TALENT protocol, we use accuracy as the primary metric for classification tasks and RMSE for regression tasks. For aggregating results across datasets, we compute: 

- **Improvability** : The relative error gap to the best method, using 1 _−_ accuracy (classification) and RMSE (regression). 

- **Elo** : Pairwise comparison-based rating using accuracy (classification) and negative RMSE (regression). 

- **Average rank** : Mean rank across datasets based on accuracy (classification) or RMSE (regression). 

TALENT also provides supplementary metrics including log-loss and AUC for classification, and MAE and _R_<sup>2</sup> for regression. In the following subsections, we present detailed results stratified by task type and dataset characteristics. 

Note that, for a fair comparison, we exclude the development datasets from the main paper used for the development of TabICLv2. 

#### **K.2. Results on all datasets** 

TabICLv2 achieves the best average rank of 4.66, outperforming RealTabPFN-2.5 (5.11) and TabPFN-2.5 (5.45). The pairwise win rates further confirm the dominance of TabICLv2: 62% against RealTabPFN-2.5 and 65% against TabPFN-2.5. 

On TALENT, we evaluate both RealTabPFN-2.5 (fine-tuned on real data) and TabPFN-2.5 (not fine-tuned). RealTabPFN-2.5 outperforms TabPFN-2.5 (5.11 vs. 5.45), demonstrating that fine-tuning on real-world data provides measurable benefits. Note that TabArena does not report results for TabPFN-2.5, making TALENT a valuable benchmark for isolating the effect of fine-tuning on TabPFN-2.5. 

TabICLv2 substantially outperforms other tabular foundation models. LimiX and TabPFNv2 achieve average ranks of 8.34 and 8.82, respectively, nearly twice that of TabICLv2. 

56 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
25 20 15 10 5<br>SwitchTab [26.97] [4.66] TabICLv2<br>TabNet [25.67] [5.11] RealTabPFN-2.5<br>GrowNet [24.53] [5.45] TabPFN-2.5<br>TabTransformer [23.76] [8.34] LimiX<br>Linear [23.04] [8.82] TabPFNv2<br>DANets [22.82] [10.71] TabICL<br>KNN [21.77] [11.66] ModernNCA<br>NODE [19.44] [11.66] CatBoost<br>TANGOS [19.02] [11.88] RealMLP<br>SNN [18.55] [12.56] TabR<br>PTaRL [17.94] [12.81] LightGBM<br>ExcelFormer [17.56] [13.68] XGBoost<br>MLP [17.50] [15.30] FT-Transformer<br>AutoInt [17.39] [15.77] MLP-PLR<br>ResNet [17.33] [17.09] RandomForest<br>DCNv2 [17.22]<br><!-- End of picture text -->

_Figure K.1._ **Critical difference diagram on the TALENT benchmark.** 



<!-- Start of picture text -->
TabICLv2 62 65 72 75 83 81 83 86 81 86 87 90 89 89 91 93 95 93 93<br>RealTabPFN-2.5 38 58 67 76 74 77 80 87 83 85 84 90 89 88 92 96 94 93 91<br>TabPFN-2.5 35 42 63 74 72 77 78 86 82 85 84 90 89 87 92 95 93 93 90<br>LimiX 28 33 37 57 59 67 66 70 71 73 74 76 78 84 81 83 82 83 88<br>TabPFNv2 25 24 26 43 57 64 65 70 69 71 70 79 79 78 82 82 81 84 85<br>TabICL 17 26 28 41 43 57 52 61 52 62 58 70 69 71 73 73 75 77 85<br>ModernNCA 19 23 23 33 36 43 53 54 54 54 56 66 72 68 75 76 75 76 81<br>CatBoost 17 20 22 34 35 48 47 56 56 56 66 66 67 74 71 73 73 73 84<br>RealMLP 14 13 14 30 30 39 46 44 48 48 55 67 70 63 79 78 79 79 81<br>LightGBM 19 17 18 29 31 48 46 44 52 50 55 59 61 74 67 70 69 66 80<br>TabR 14 15 15 27 29 38 46 44 52 50 52 66 68 63 75 72 74 77 77<br>XGBoost 13 16 16 26 30 42 44 34 45 45 48 56 58 72 61 65 64 65 81<br>FT-Transformer 10 10 10 24 21 30 34 34 33 41 34 44 49 57 66 65 61 60 71<br>MLP-PLR 11 11 11 22 21 31 28 33 30 39 32 42 51 54 59 60 59 60 72<br>RandomForest 11 12 13 16 22 29 32 26 37 26 37 28 43 46 48 48 51 50 72<br>AutoInt 9 8 8 19 18 27 25 29 21 33 25 39 34 41 52 47 48 52 69<br>ExcelFormer 7 4 5 17 18 27 24 27 22 30 28 35 35 40 52 53 54 52 69<br>ResNet 5 6 7 18 19 25 25 27 21 31 26 36 39 41 49 52 46 58 68<br>MLP 7 7 7 17 16 23 24 27 21 34 23 35 40 40 50 48 48 42 72<br>KNN 7 9 10 12 15 15 19 16 19 20 23 19 29 28 28 31 31 32 28<br><!-- End of picture text -->



_Figure K.2._ **Winrate matrix on the TALENT benchmark.** 

57 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
MLP<br>Pareto Front ExcelFormer<br>ResNet<br>25% RandomForest<br>AutoInt<br>MLP-PLR<br>FT-Transformer<br>20% XGBoost<br>LightGBM<br>CatBoost<br>TabR<br>15% RealMLP<br>ModernNCA<br>TabICL<br>TabPFNv2<br>10% LimiX<br>TabPFN-2.5<br>RealTabPFN-2.5<br>TabICLv2<br>10 0 10 1<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Improvability (%)<br><!-- End of picture text -->

_Figure K.3._ **Pareto front of improvability and inference time on the TALENT benchmark.** 



<!-- Start of picture text -->
TabICLv2<br>RealTabPFN-2.5<br>1400 TabPFN-2.5<br>LimiX<br>TabPFNv2<br>1300 TabICL<br>ModernNCA<br>CatBoost<br>1200<br>RealMLP<br>TabR<br>1100 LightGBM<br>XGBoost<br>FT-Transformer<br>1000 MLP-PLR<br>RandomForest<br>ResNet<br>900 AutoInt<br>Pareto Front MLP<br>ExcelFormer<br>800<br>10 − 1 10 0 10 1 10 2 KNN<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Elo<br><!-- End of picture text -->

_Figure K.4._ **Pareto front of Elo and inference time on the TALENT benchmark.** 

58 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **K.3. Results on binary classification datasets** 

For binary classification datasets, TabICLv2 (4.98) is essentially tied with RealTabPFN-2.5 (4.82) on accuracy, both outperforming TabPFN-2.5 (5.28), TabICL (8.48), and LimiX (8.61). However, on AUC and log-loss, TabICLv2 achieves a clear lead: 

- **AUC** : TabICLv2 (3.31) vs. RealTabPFN-2.5 (4.62) vs. TabPFN-2.5 (5.45) 

- **log-loss** : TabICLv2 (2.83) vs. RealTabPFN-2.5 (3.78) vs. TabPFN-2.5 (4.31) 

The gap between accuracy and probabilistic metrics (AUC, log-loss) reveals important differences in model behavior. Accuracy only evaluates predictions at a fixed decision threshold, whereas AUC measures ranking quality across all thresholds and log-loss evaluates probability calibration. The fact that TabICLv2 outperforms RealTabPFN-2.5 on AUC and log-loss while matching on accuracy suggests that TabICLv2 produces better probability estimates. This is valuable in practice, where better probabilities enable reliable uncertainty quantification and informed decision-making beyond simple class predictions. 

59 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
25 20 15 10 5<br>TabNet [26.62] [4.82] RealTabPFN-2.5<br>SwitchTab [26.21] [4.98] TabICLv2<br>GrowNet [23.81] [5.28] TabPFN-2.5<br>KNN [22.96] [8.48] TabICL<br>Linear [21.69] [8.61] LimiX<br>TabTransformer [21.16] [8.92] TabPFNv2<br>DANets [19.97] [12.47] CatBoost<br>NODE [19.44] [12.81] TabR<br>TANGOS [19.02] [12.87] LightGBM<br>ExcelFormer [18.74] [12.95] ModernNCA<br>SNN [18.46] [13.45] RealMLP<br>PTaRL [18.33] [14.16] XGBoost<br>ResNet [17.97] [15.74] FT-Transformer<br>RandomForest [17.88] [16.26] MLP-PLR<br>MLP [17.47] [17.22] DCNv2<br>AutoInt [17.26]<br>(a)  Accuracy<br>25 20 15 10 5<br>TabNet [27.96] [3.31] TabICLv2<br>SwitchTab [25.91] [4.62] RealTabPFN-2.5<br>KNN [24.54] [5.45] TabPFN-2.5<br>GrowNet [23.52] [6.34] TabICL<br>TabTransformer [21.76] [7.14] LimiX<br>Linear [21.27] [8.07] TabPFNv2<br>DANets [20.68] [12.10] CatBoost<br>NODE [19.77] [12.73] TabR<br>TANGOS [19.51] [12.75] ModernNCA<br>PTaRL [19.32] [13.75] LightGBM<br>DCNv2 [18.33] [14.02] XGBoost<br>ExcelFormer [18.31] [15.40] FT-Transformer<br>SNN [17.92] [15.65] RealMLP<br>ResNet [17.86] [16.05] MLP-PLR<br>MLP [17.54] [16.90] RandomForest<br>AutoInt [17.50]<br>(b)  AUC<br>25 20 15 10 5<br>KNN [27.23] [2.83] TabICLv2<br>TabNet [24.44] [3.78] RealTabPFN-2.5<br>SwitchTab [23.81] [4.31] TabPFN-2.5<br>DANets [22.41] [5.90] TabICL<br>TabTransformer [22.04] [6.62] LimiX<br>GrowNet [21.94] [6.86] TabPFNv2<br>PTaRL [20.04] [11.27] CatBoost<br>NODE [19.81] [13.53] ModernNCA<br>MLP [19.80] [13.76] TabR<br>SNN [19.67] [14.26] LightGBM<br>DCNv2 [19.41] [14.48] FT-Transformer<br>Linear [19.30] [16.18] AutoInt<br>MLP-PLR [18.54] [16.23] ExcelFormer<br>TANGOS [18.18] [16.58] XGBoost<br>RealMLP [18.07] [16.90] RandomForest<br>ResNet [17.82]<br>(c)  Log-Loss<br><!-- End of picture text -->

_Figure K.5._ **Critical difference diagram on binary classification datasets of the TALENT benchmark.** 

60 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
ExcelFormer<br>25% Pareto Front MLP<br>RandomForest<br>MLP-PLR<br>AutoInt<br>20% ResNet<br>FT-Transformer<br>XGBoost<br>RealMLP<br>15% LightGBM<br>CatBoost<br>TabR<br>ModernNCA<br>10% TabPFNv2<br>TabICL<br>LimiX<br>TabPFN-2.5<br>5% RealTabPFN-2.5<br>TabICLv2<br>10 0 10 1<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Improvability (%)<br><!-- End of picture text -->

_Figure K.6._ **Pareto front of improvability and inference time on binary classification datasets of the TALENT benchmark.** 



<!-- Start of picture text -->
1500 RealTabPFN-2.5<br>TabICLv2<br>1400 TabPFN-2.5<br>TabICL<br>LimiX<br>1300 TabPFNv2<br>CatBoost<br>TabR<br>1200<br>LightGBM<br>ModernNCA<br>1100 RealMLP<br>XGBoost<br>FT-Transformer<br>1000 MLP-PLR<br>AutoInt<br>MLP<br>900<br>RandomForest<br>Pareto Front ResNet<br>800 ExcelFormer<br>10 − 1 10 0 10 1 KNN<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Elo<br><!-- End of picture text -->

_Figure K.7._ **Pareto front of Elo and inference time on binary classification datasets of the TALENT benchmark.** 

61 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **K.4. Results on multiclass classification datasets (** _≤_ **10 classes)** 

Unlike binary classification where TabICLv2 and RealTabPFN-2.5 are comparable on accuracy, TabICLv2 achieves clear superiority on multiclass tasks across all evaluation metrics: 

- **Accuracy** : TabICLv2 (4.58) vs. RealTabPFN-2.5 (5.64) 

- **AUC** : TabICLv2 (3.38) vs. RealTabPFN-2.5 (5.20) 

- **Log-loss** : TabICLv2 (2.67) vs. RealTabPFN-2.5 (4.48) 

62 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
25 20 15 10 5<br>TabNet [27.27] [4.58] TabICLv2<br>SwitchTab [26.69] [5.64] RealTabPFN-2.5<br>GrowNet [25.46] [6.24] TabPFN-2.5<br>NODE [23.68] [8.07] TabICL<br>TabTransformer [22.98] [9.03] LimiX<br>Linear [22.24] [9.38] TabPFNv2<br>KNN [21.14] [10.12] ModernNCA<br>PTaRL [20.73] [10.26] RealMLP<br>DANets [19.93] [10.80] TabR<br>TANGOS [19.23] [12.43] CatBoost<br>ExcelFormer [18.64] [13.93] LightGBM<br>AutoInt [18.36] [14.63] XGBoost<br>RandomForest [18.21] [14.90] ResNet<br>SNN [17.38] [15.37] FT-Transformer<br>MLP [16.79] [15.93] MLP-PLR<br>DCNv2 [15.97]<br>(a)  Accuracy<br>25 20 15 10 5<br>TabNet [27.10] [3.38] TabICLv2<br>SwitchTab [26.67] [5.20] RealTabPFN-2.5<br>KNN [24.77] [5.67] TabPFN-2.5<br>GrowNet [24.68] [7.36] TabICL<br>NODE [23.13] [8.18] TabPFNv2<br>Linear [22.78] [8.87] LimiX<br>TabTransformer [21.16] [10.50] ModernNCA<br>PTaRL [20.40] [11.42] CatBoost<br>DANets [19.72] [13.72] RealMLP<br>RandomForest [18.17] [14.23] XGBoost<br>TANGOS [18.02] [14.32] LightGBM<br>ExcelFormer [17.53] [14.68] TabR<br>MLP [17.08] [15.22] MLP-PLR<br>DCNv2 [16.83] [15.23] ResNet<br>AutoInt [16.78] [16.52] FT-Transformer<br>SNN [16.70]<br>(b)  AUC<br>25 20 15 10 5<br>KNN [27.10] [2.67] TabICLv2<br>TabNet [24.15] [4.48] RealTabPFN-2.5<br>NODE [23.60] [4.78] TabPFN-2.5<br>SwitchTab [23.42] [5.83] TabICL<br>GrowNet [22.79] [7.52] TabPFNv2<br>TabTransformer [21.51] [8.07] LimiX<br>DANets [21.48] [9.92] CatBoost<br>Linear [21.12] [11.82] ModernNCA<br>PTaRL [20.03] [12.90] TabR<br>RandomForest [18.77] [14.25] FT-Transformer<br>MLP [18.77] [15.73] LightGBM<br>SNN [18.77] [15.80] ResNet<br>DCNv2 [18.42] [15.93] AutoInt<br>XGBoost [18.15] [16.20] TANGOS<br>MLP-PLR [18.02] [16.38] ExcelFormer<br>RealMLP [17.63]<br>(c)  Log-Loss<br><!-- End of picture text -->

_Figure K.8._ **Critical difference diagram on multiclass classification datasets (** _≤_ **10 classes) of the TALENT benchmark.** 

63 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
MLP<br>Pareto Front AutoInt<br>RandomForest<br>35%<br>ExcelFormer<br>FT-Transformer<br>ResNet<br>30%<br>MLP-PLR<br>XGBoost<br>25% LightGBM<br>CatBoost<br>TabR<br>ModernNCA<br>20%<br>RealMLP<br>TabICL<br>LimiX<br>15% TabPFNv2<br>TabPFN-2.5<br>RealTabPFN-2.5<br>10% TabICLv2<br>10 0 10 1<br>Inference time per 1K samples (s) (median)<br>TabICLv2<br>1500<br>RealTabPFN-2.5<br>TabPFN-2.5<br>1400 TabICL<br>LimiX<br>TabPFNv2<br>1300 ModernNCA<br>RealMLP<br>TabR<br>1200 CatBoost<br>LightGBM<br>XGBoost<br>1100<br>ResNet<br>FT-Transformer<br>1000 MLP-PLR<br>MLP<br>RandomForest<br>900 Pareto Front AutoInt<br>ExcelFormer<br>10 − 1 10 0 10 1 10 2 KNN<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Optimal<br>Improvability (%)<br>Elo<br><!-- End of picture text -->

_Figure K.9._ **Pareto front of improvability and inference time on multiclass classification datasets of the TALENT benchmark.** 

_Figure K.10._ **Pareto front of Elo and inference time on multiclass classification datasets (** _≤_ **10 classes) of the TALENT benchmark.** 

64 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **K.5. Results on multiclass classification datasets (** _>_ **10 classes)** 

This section presents results on 12 multiclass classification datasets with more than 10 classes in TALENT. TabICLv2 clearly outperforms RealTabPFN-2.5 on many-class classification. Here, ECOC refers to the error-correcting output codes wrapper from TabPFNv2 (Hollmann et al., 2025). TabICLv2 achieves strong performance both with the ECOC wrapper and with its native many-class handling via mixed-radix ensembling. Both variants outperform RealTabPFN-2.5-ECOC. 

65 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
25 20 15 10 5<br>TabNet [25.75] [2.92] TabICLv2-ECOC<br>NODE [25.75] [3.38] TabICLv2<br>SwitchTab [25.25] [5.92] ModernNCA<br>GrowNet [24.92] [6.08] RealMLP<br>Linear [22.17] [6.21] RealTabPFN-2.5-ECOC<br>RandomForest [21.58] [6.92] TabICL<br>TabTransformer [21.17] [8.00] TabR<br>KNN [21.00] [8.50] ResNet<br>PTaRL [19.92] [10.62] FT-Transformer<br>LightGBM [18.67] [10.71] MLP-PLR<br>ExcelFormer [17.92] [11.83] MLP<br>XGBoost [17.83] [13.25] DCNv2<br>TANGOS [17.83] [14.25] SNN<br>DANets [17.75] [14.33] AutoInt<br>CatBoost [14.58]<br>(a)  Accuracy<br>25 20 15 10 5<br>GrowNet [25.42] [2.46] TabICLv2-ECOC<br>TabNet [25.33] [2.88] TabICLv2<br>KNN [25.25] [5.71] TabICL<br>NODE [24.33] [7.12] RealTabPFN-2.5-ECOC<br>SwitchTab [23.08] [7.25] ModernNCA<br>Linear [21.92] [9.58] MLP-PLR<br>RandomForest [20.75] [9.67] ResNet<br>PTaRL [19.75] [10.08] FT-Transformer<br>LightGBM [19.58] [10.92] MLP<br>XGBoost [19.25] [10.92] TabR<br>TabTransformer [18.50] [12.75] AutoInt<br>DANets [15.50] [13.83] SNN<br>TANGOS [15.33] [14.25] RealMLP<br>ExcelFormer [14.75] [14.33] DCNv2<br>CatBoost [14.50]<br>(b)  AUC<br>25 20 15 10 5<br>KNN [26.83] [1.83] TabICLv2-ECOC<br>NODE [24.00] [2.00] TabICLv2<br>TabNet [23.00] [5.58] TabICL<br>GrowNet [23.00] [7.58] RealTabPFN-2.5-ECOC<br>RandomForest [21.58] [8.83] TabR<br>Linear [21.25] [9.17] ModernNCA<br>SwitchTab [21.25] [9.42] FT-Transformer<br>TabTransformer [20.58] [10.50] ResNet<br>DANets [20.00] [11.67] MLP<br>PTaRL [18.83] [11.75] AutoInt<br>XGBoost [18.33] [11.92] MLP-PLR<br>LightGBM [18.25] [13.00] CatBoost<br>SNN [17.67] [13.08] RealMLP<br>DCNv2 [15.67] [14.17] ExcelFormer<br>TANGOS [14.25]<br>(c)  Log-Loss<br><!-- End of picture text -->

_Figure K.11._ **Critical difference diagram on multiclass classification datasets (** _>_ **10 classes) of the TALENT benchmark.** 

66 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **K.6. Results on regression datasets** 

TabICLv2 outperforms TabPFN-2.5 on RMSE and _R_<sup>2</sup> , while TabPFN-2.5 has a slight edge on MAE. RMSE penalizes large errors, making it sensitive to outliers and tail behavior, while MAE treats all errors equally. The quantile regression of TabICLv2, trained with pinball loss across 999 quantiles, is designed to capture the full conditional distribution rather than optimizing for a single point estimate. This distributional focus leads to better performance on RMSE and _R_<sup>2</sup> , which reward accurate modeling of variance and extreme values. The slight disadvantage on MAE suggests that the bin-based approach of TabPFN-2.5 may be marginally better optimized for median prediction, though the difference is small (4.43 vs. 4.63). 

67 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
25 20 15 10 5<br>DANets [27.43] [4.21] TabICLv2<br>SwitchTab [27.13] [5.01] TabPFN-2.5<br>TabTransformer [26.61] [5.01] RealTabPFN-2.5<br>Linear [24.36] [7.34] LimiX<br>GrowNet [23.92] [8.09] TabPFNv2<br>TabNet [22.51] [9.87] CatBoost<br>KNN [19.89] [10.57] RealMLP<br>SNN [18.85] [10.67] ModernNCA<br>TANGOS [18.26] [11.71] LightGBM<br>ResNet [17.60] [12.18] XGBoost<br>DCNv2 [17.48] [12.96] TabR<br>MLP [17.46] [14.19] FT-Transformer<br>AutoInt [16.43] [14.56] MLP-PLR<br>NODE [16.00] [14.82] RandomForest<br>PTaRL [15.06] [14.84] ExcelFormer<br>(a)  RMSE<br><!-- End of picture text -->



<!-- Start of picture text -->
25 20 15 10 5<br>DANets [27.42] [4.20] TabICLv2<br>SwitchTab [27.12] [5.02] TabPFN-2.5<br>TabTransformer [26.57] [5.02] RealTabPFN-2.5<br>Linear [24.34] [7.29] LimiX<br>GrowNet [23.90] [8.03] TabPFNv2<br>TabNet [22.56] [9.84] CatBoost<br>KNN [19.85] [10.62] RealMLP<br>SNN [18.83] [10.73] ModernNCA<br>TANGOS [18.25] [11.65] LightGBM<br>ResNet [17.64] [12.18] XGBoost<br>MLP [17.49] [12.98] TabR<br>DCNv2 [17.44] [14.21] FT-Transformer<br>AutoInt [16.43] [14.58] MLP-PLR<br>NODE [15.97] [14.76] RandomForest<br>PTaRL [15.18] [14.89] ExcelFormer<br><!-- End of picture text -->



<!-- Start of picture text -->
(b)  R2<br><!-- End of picture text -->



<!-- Start of picture text -->
25 20 15 10 5<br>TabTransformer [27.03] [4.43] TabPFN-2.5<br>DANets [26.63] [4.43] RealTabPFN-2.5<br>SwitchTab [26.19] [4.63] TabICLv2<br>Linear [25.33] [7.65] TabPFNv2<br>GrowNet [24.02] [8.11] LimiX<br>TabNet [23.88] [10.03] TabR<br>KNN [19.78] [10.99] RealMLP<br>PTaRL [18.16] [11.42] ModernNCA<br>SNN [17.78] [11.58] MLP-PLR<br>NODE [17.62] [12.38] CatBoost<br>TANGOS [17.07] [13.04] FT-Transformer<br>ResNet [16.52] [14.35] XGBoost<br>DCNv2 [16.40] [14.53] AutoInt<br>RandomForest [16.12] [14.72] LightGBM<br>ExcelFormer [15.16] [15.02] MLP<br>(c)  MAE<br><!-- End of picture text -->

_Figure K.12._ **Critical difference diagram on regression datasets of the TALENT benchmark.** 

68 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
MLP<br>ResNet<br>25%<br>AutoInt<br>TabICL<br>22% RandomForest<br>ExcelFormer<br>20% MLP-PLR<br>FT-Transformer<br>XGBoost<br>17%<br>Pareto Front ModernNCA<br>LightGBM<br>15% TabR<br>CatBoost<br>12% RealMLP<br>TabPFNv2<br>10% LimiX<br>RealTabPFN-2.5<br>TabPFN-2.5<br>7%<br>TabICLv2<br>10 0 10 1 10 2<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Improvability (%)<br><!-- End of picture text -->

_Figure K.13._ **Pareto front of improvability and inference time on regression classification datasets of the TALENT benchmark.** 



<!-- Start of picture text -->
TabICLv2<br>TabPFN-2.5<br>1400 RealTabPFN-2.5<br>LimiX<br>TabPFNv2<br>1300<br>CatBoost<br>RealMLP<br>1200 ModernNCA<br>LightGBM<br>XGBoost<br>1100 TabR<br>FT-Transformer<br>1000 MLP-PLR<br>TabICL<br>RandomForest<br>900 ExcelFormer<br>AutoInt<br>800 Pareto Front MLP<br>ResNet<br>10 − 1 10 0 10 1 10 2 KNN<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Elo<br><!-- End of picture text -->

_Figure K.14._ **Pareto front of Elo and inference time on regression datasets of the TALENT benchmark.** 

69 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **K.7. Results on small datasets with less than 10K samples** 

TabICLv2 and RealTabPFN-2.5 achieve comparable performance on small datasets, the regime where tabular foundation models are originally designed to excel. 



<!-- Start of picture text -->
25 20 15 10 5<br>TabNet [26.69] [4.89] TabICLv2<br>SwitchTab [26.23] [4.91] RealTabPFN-2.5<br>GrowNet [23.80] [5.19] TabPFN-2.5<br>TabTransformer [22.94] [6.46] LimiX<br>DANets [22.47] [7.26] TabPFNv2<br>Linear [21.57] [9.77] TabICL<br>KNN [20.68] [12.22] ModernNCA<br>NODE [19.31] [12.36] CatBoost<br>ExcelFormer [18.66] [12.73] LightGBM<br>SNN [18.59] [13.41] RealMLP<br>PTaRL [18.49] [13.91] XGBoost<br>AutoInt [18.24] [14.07] TabR<br>DCNv2 [18.07] [15.41] RandomForest<br>TANGOS [18.07] [16.56] FT-Transformer<br>MLP [17.91] [17.54] MLP-PLR<br>ResNet [17.59]<br>Figure K.15. Critical difference diagram on small datasets of the TALENT benchmark.<br>MLP<br>30% Pareto Front ExcelFormer<br>AutoInt<br>ResNet<br>25% MLP-PLR<br>FT-Transformer<br>RandomForest<br>TabR<br>20% XGBoost<br>LightGBM<br>CatBoost<br>RealMLP<br>15%<br>ModernNCA<br>TabICL<br>TabPFNv2<br>10% LimiX<br>TabPFN-2.5<br>RealTabPFN-2.5<br>TabICLv2<br>10 0 10 1<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Improvability (%)<br><!-- End of picture text -->

_Figure K.16._ **Pareto front of improvability and inference time on small datasets of the TALENT benchmark.** 

70 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
TabICLv2<br>1400 RealTabPFN-2.5<br>TabPFN-2.5<br>LimiX<br>1300<br>TabPFNv2<br>TabICL<br>ModernNCA<br>1200<br>CatBoost<br>LightGBM<br>RealMLP<br>1100<br>XGBoost<br>TabR<br>1000 RandomForest<br>FT-Transformer<br>MLP-PLR<br>900 ResNet<br>MLP<br>Pareto Front AutoInt<br>800 ExcelFormer<br>10 − 1 10 0 10 1 KNN<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Elo<br><!-- End of picture text -->

_Figure K.17._ **Pareto front of Elo and inference time on small datasets of the TALENT benchmark.** 

#### **K.8. Results on large datasets with more than 10K samples** 

On large datasets, TabICLv2 demonstrates a clear advantage over both RealTabPFN-2.5 and TabPFN-2.5. 



<!-- Start of picture text -->
25 20 15 10 5<br>SwitchTab [28.33] [4.25] TabICLv2<br>GrowNet [25.88] [5.49] RealTabPFN-2.5<br>Linear [25.76] [5.92] TabPFN-2.5<br>TabTransformer [25.27] [9.06] RealMLP<br>KNN [23.80] [9.76] TabR<br>TabNet [23.78] [10.37] CatBoost<br>DANets [23.45] [10.62] ModernNCA<br>TANGOS [20.77] [11.71] TabPFNv2<br>RandomForest [20.18] [11.79] LimiX<br>NODE [19.68] [12.45] TabICL<br>SNN [18.47] [12.50] MLP-PLR<br>PTaRL [16.93] [12.96] LightGBM<br>ResNet [16.87] [12.97] FT-Transformer<br>MLP [16.74] [13.25] XGBoost<br>AutoInt [15.83] [15.53] ExcelFormer<br>DCNv2 [15.63]<br><!-- End of picture text -->

_Figure K.18._ **Critical difference diagram on large datasets of the TALENT benchmark.** 

71 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
RandomForest<br>Pareto Front MLP<br>ResNet<br>25% AutoInt<br>ExcelFormer<br>FT-Transformer<br>LightGBM<br>20% XGBoost<br>MLP-PLR<br>TabICL<br>LimiX<br>15% TabPFNv2<br>CatBoost<br>ModernNCA<br>TabR<br>10% RealMLP<br>TabPFN-2.5<br>RealTabPFN-2.5<br>TabICLv2<br>10 0 10 1 10 2<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Improvability (%)<br><!-- End of picture text -->

_Figure K.19._ **Pareto front of improvability and inference time on large datasets of the TALENT benchmark.** 



<!-- Start of picture text -->
1600 TabICLv2<br>RealTabPFN-2.5<br>TabPFN-2.5<br>1500<br>RealMLP<br>TabR<br>1400 CatBoost<br>ModernNCA<br>1300 TabPFNv2<br>LimiX<br>TabICL<br>1200<br>MLP-PLR<br>LightGBM<br>1100 FT-Transformer<br>XGBoost<br>1000 ExcelFormer<br>AutoInt<br>900 MLP<br>Pareto Front ResNet<br>RandomForest<br>800<br>10 0 10 1 10 2 KNN<br>Inference time per 1K samples (s) (median)<br>Optimal<br>Elo<br><!-- End of picture text -->

_Figure K.20._ **Pareto front of Elo and inference time on large datasets of the TALENT benchmark.** 

72 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 

#### **K.9. Model rankings with respect to meta-features** 



<!-- Start of picture text -->
30 TabICLv2 30 TabICLv2<br>TabPFNv2 TabPFNv2<br>25 RealTabPFN-2.5 25 RealTabPFN-2.5<br>RealMLP RealMLP<br>20 CatBoost 20 CatBoost<br>15<br>15<br>10<br>10<br>5<br>5<br>0<br>0<br>10 3 10 4 10 5 10 1 10 2<br>Number of samples Number of features<br>(a) (b)<br>Rank Rank<br>← ←<br><!-- End of picture text -->

_Figure K.21._ **Model rankings with respect to meta-features across all datasets of the TALENT benchmark.** 

73 

**TabICLv2 : A better, faster, scalable, and open tabular foundation model** 



<!-- Start of picture text -->
30 TabICL 30 TabICL<br>TabICLv2 TabICLv2<br>25 TabPFNv2 25 TabPFNv2<br>RealTabPFN-2.5 RealTabPFN-2.5<br>20 RealMLP 20 RealMLP<br>CatBoost CatBoost<br>15<br>15<br>10<br>10<br>5<br>5<br>0<br>0<br>10 3 10 4 10 5 10 1 10 2<br>Number of samples Number of features<br>(a) (b)<br>30 TabICL 30 TabICL<br>TabICLv2 TabICLv2<br>25 TabPFNv2 25 TabPFNv2<br>RealTabPFN-2.5 RealTabPFN-2.5<br>20 RealMLP 20 RealMLP<br>CatBoost CatBoost<br>15 15<br>10 10<br>5 5<br>0 0<br>2 3 4 5 6 7 8 9 10 0 . 0 0 . 2 0 . 4 0 . 6 0 . 8 1 . 0<br>Number of classes Ratio of categorical features<br>(c) (d)<br>Figure K.22. Model rankings with respect to meta-features across classification datasets of the TALENT benchmark.<br>30 30<br>TabICLv2 TabICLv2<br>TabPFNv2 TabPFNv2<br>25 25<br>RealTabPFN-2.5 RealTabPFN-2.5<br>RealMLP RealMLP<br>20 20<br>CatBoost CatBoost<br>15 15<br>10 10<br>5 5<br>0<br>0<br>10 3 10 4 10 5 10 1 10 2<br>Number of samples Number of features<br>(a) (b)<br>Rank Rank<br>← ←<br>Rank Rank<br>← ←<br>Rank Rank<br>← ←<br><!-- End of picture text -->

_Figure K.23._ **Model rankings with respect to meta-features across regression datasets of the TALENT benchmark.** 

74 


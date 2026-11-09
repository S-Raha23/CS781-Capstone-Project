# CS-781 Capstone: In-House Legal Intelligence System

Soubhagya Raha (231025), IIT Kanpur.

This repository holds the code for the seven phases of the CS-781 capstone. One legal language model is trained from scratch in Phase 1, and its checkpoints are the starting point for six downstream legal NLP tasks. No other pretrained language model or external LLM API is used in any model.

## Official results

Scores are the Codabench results of the submissions selected as final.

| Phase | Task | Final submission | Official score |
|---|---|---|---|
| 1 | Legal language model pretraining | `submission_decoder_continued_decoding-greedy_ngram3.zip` | 0.8346 |
| 2 | Legal NER | `B_greedy__piece-viterbi__prop0.1.zip` | 0.5419 |
| 3 | Rhetorical role prediction | `decoder_lora__mean__lam0.5__tau0.5.zip` | 0.4922 |
| 4 | Court judgment prediction | `decoder_partial__all__last__thr0.4.zip` | 0.7198 |
| 5 | Legal statute identification | `greedy__all__max__perlabel.zip` | 0.3195 |
| 6 | Abstractive summarisation | `decoder_lora__head__greedy_nr3_w300.zip` | 0.8379 |
| 7 | Prior case retrieval | `ens5.zip` | 0.3585 |

## Repository layout

```
code/
  phase1/  phase1-data-prep.ipynb    tokenizer and corpus (notebook A)
           phase1-train.ipynb        decoder, prefix, encdec, swa (notebook B)
           phase1-train-extra.ipynb  diffusion, moe (notebook C)
           phase1-continue.ipynb     continued decoder training, decoding sweep (notebook D)
  phase2/  phase2-ner.ipynb          NER pipeline (notebook E)
           phase2-ner-v2.ipynb       Viterbi, propagation, ensembles, final zip (notebook F)
  phase3/  phase3-rr.ipynb           (notebook G)
  phase4/  phase4-cjpe.ipynb         first run (notebook H)
           phase4-cjpe-part2.ipynb   decoder runs, final model (notebook H2)
  phase5/  phase5-lsi.ipynb          (notebook I)
  phase6/  phase6-summ.ipynb         (notebook J)
  phase7/  phase7-pcr.ipynb          (notebook K)
```

The `validation_index*.csv` files list the validation scores and settings of the prediction sets each notebook's lab produced. They are **not** Codabench scores. `phase4/validation_index_first_run.csv` comes from the first Phase 4 run and has no `decoder_partial` rows. Phase 6 has no index file here.

Checkpoints and datasets are not stored in this repository because of their size.

## How to run

Every notebook was run on Kaggle with **GPU T4 x2** and Internet on, with the Kaggle secret `Capstone_Phase_1` attached as the Hugging Face token. Data comes from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`.

Each notebook writes its modules with `%%writefile` cells, so it is self-contained apart from its Kaggle inputs. Run it with **Save & Run All**. Run them in this order:

1. `phase1-data-prep` (CPU, about 1 hour). Produces `bpe.json`, `bpe_tok.py`, `train.bin`, `val.bin`, `val_prompts.json`, `val_truth.json`.
2. `phase1-train`, once with `SESSION_ARCHS = ["decoder", "prefix"]` and once with `["encdec", "swa"]`. Each session takes about 5.5 hours. Produces `runs/<arch>/best.pt`.
3. `phase1-train-extra` for `diffusion` and `moe`, about 5 to 5.5 hours.
4. `phase1-continue`, with the session 1 output of `phase1-train` attached. Produces `continued/best_continued.pt` and the final Phase 1 zip.
5. Any later phase, with these outputs attached as inputs. Each notebook's first cell lists its inputs.

Later phases find checkpoints under `/kaggle/input` at these paths:

| Architecture | Checkpoint path |
|---|---|
| decoder | `continued/best_continued.pt` |
| prefix | `prefix/best.pt` |
| encdec | `encdec/best.pt` |
| swa | `swa/best.pt` |
| diffusion | `diffusion/best.pt` |
| moe | `moe/best.pt` |

Each later phase has the same cell structure:

| Cell group | Purpose |
|---|---|
| N1 to N3 | Login and data download, GPU check, configuration |
| N4 | Pipeline module (`*_common.py`) |
| N5 | Phase 1 model code (`models.py`, `models_x.py`) |
| N6 | Fine-tuning methods (`*_methods.py`) |
| N7 | Training script, lab, official evaluator |
| N8 to N10 | Data check, CPU smoke test, training |
| N11 | Lab: scores prediction sets and writes zips and `submissions/index.csv` |
| N12 | Checks zip format and confirms lab scores with the official evaluator |
| N13 | Tables and error-analysis material |

Download the zips from `/kaggle/working/submissions/`.

## Phase 1: Legal language model

**Task.** Train a 50M to 300M parameter language model from scratch and continue a 512-word prompt with the next 128 words. The score is BERTScore F1.

**Data.** Words kept per source, after filtering documents under 200 words or with more than 1% Devanagari characters:

| Source | Target training words |
|---|---|
| Supreme Court | 210M |
| High Courts | 110M |
| Tribunals | 60M |
| Acts | 25M |

**Tokenizer.** A custom byte-pair-encoding tokenizer trained on a 5M-word sample that follows the source mix. A coverage document gives every UTF-8 byte a base token. Vocabulary size is 16,000.

**Architectures.** All use `d_model = 672`, 8 heads, GELU, pre-norm layers, feed-forward width `4 * d_model`, dropout 0.1, learned position embeddings and weight tying.

| Name | Class | Structure |
|---|---|---|
| `decoder` | `DecoderLM` | 8 causal layers, 1024 tokens, 54.86M parameters |
| `prefix` | `PrefixLM` | 8 layers, bidirectional over the prefix, causal after it |
| `encdec` | `EncDecLM` | 4 encoder and 4 decoder layers, 768 source and 384 target tokens |
| `swa` | `SlidingWindowLM` | 8 layers, sliding-window attention (window 256) with ALiBi bias |
| `diffusion` | `LexDiffusionLM` | 8 layers, masked-diffusion language model |
| `moe` | `LexMoE` | 8 causal layers, feed-forward replaced by 4 routed experts (`d_ff` 2688), load-balancing coefficient 0.01 |

**Training.** AdamW with learning rate 3e-4, weight decay 0.1, betas (0.9, 0.999), 500 warm-up steps and cosine decay over the wall-clock budget. Block size 1024, batch 8 with accumulation 2, gradient clipping 1.0, mixed precision. The budget was 4.5 hours per model for notebook B and 4.0 hours for notebook C. The checkpoint with the best validation loss is kept.

**Final model.** The decoder checkpoint was trained further in `phase1-continue`: 10,800 steps and 176.9M more tokens, with validation loss going from 2.8738 to 2.6557.

**Decoding.** Candidate settings were compared on 287 validation prompts. The final setting `greedy_ngram3` uses temperature 0 and blocks repeated 3-grams. Generation uses a key/value cache.

| Submission | Official score |
|---|---|
| **decoder_continued (final)** | **0.8346** |
| decoder_decoding-greedy | 0.8331 |
| decoder | 0.8320 |
| swa | 0.8306 |
| moe | 0.8301 |
| prefix | 0.8262 |
| encdec | 0.8247 |
| diffusion | 0.7994 |

## Fine-tuning methods used in Phases 2 to 7

These live in each notebook's N6 cell (`*_methods.py`).

| Method | Key | Description |
|---|---|---|
| Full fine-tuning | `full` | All parameters trainable |
| Partial fine-tuning | `partial` | Only the last `n_last` layers trainable (default 2) |
| BitFit | `bitfit` | Only bias terms trainable |
| LoRA | `lora` | Low-rank additive adapters, `r = 8`, `alpha = 16` |
| LoRMA | `lorma` | Low-rank multiplicative adapters, `r = 8`, `alpha = 16` |
| rsLoRA | `rslora` | LoRA with rank-scaled factor |
| IA3 | `ia3` | Inhibit, amplify and attend scaling vectors |
| Prompt tuning | `prompt` | Trainable prompt tokens (`n_tokens = 20`) |
| Knowledge distillation | `teacher` set | `moe_full` teacher, 4-layer student |

## Phase 2: Legal NER

**Task.** Extract entity spans with exact character offsets. The score is strict macro-F1.

**Method.** Each model gets a token-classification head with BIO tags. Documents are cut into windows of 1024 tokens with stride 768. Training uses 56 documents, with 14 held out for validation. Defaults: up to 10 epochs, patience 3, learning rate 5e-5, batch 8, head dropout 0.1, weight decay 0.01.

**Post-processing** (`phase2-ner-v2`, no retraining):
- Viterbi decoding restricted to valid BIO sequences.
- `align = piece`, where tags are read at the first token of each piece, so spans end on piece boundaries.
- Document-level propagation: a string tagged in at least `min_frac` of its occurrences is tagged everywhere in that document with its majority label.
- Probability ensembles and greedy forward selection on validation.

**Final system.** Greedy ensemble of `moe_full` (full fine-tuning, 141.7M parameters) and `decoder_lora` (LoRA, 0.70M trainable of 55.6M), with `piece` alignment, Viterbi decoding and `min_frac = 0.1`. Validation macro-F1 0.5299.

| Submission | Official score |
|---|---|
| **B_greedy__piece-viterbi__prop0.1 (final)** | **0.5419** |
| B_greedy__token-viterbi__prop0.1 | 0.5171 |
| B_decoder_lora__token-viterbi | 0.5168 |
| B_moe_full__piece-viterbi__prop0.5 | 0.4604 |
| B_moe_full__piece-viterbi__prop0.1 | 0.4581 |
| B_swa_bitfit__piece-viterbi | 0.4368 |
| submission_decoder_lora | 0.3439 |
| A_kd_moe_to_decoder4__prop0.2 | 0.3121 |
| submission_moe_full | 0.3026 |
| submission_kd_moe_to_decoder4 | 0.2790 |
| submission_swa_bitfit | 0.2321 |
| submission_diffusion_full | 0.2152 |
| submission_prefix_full | 0.2007 |
| submission_encdec_lorma | 0.0419 |

## Phase 3: Rhetorical role prediction

**Task.** Assign one of 13 rhetorical roles to each sentence, across the Corporate Law and IT domains. The score is macro-F1.

**Method.** A document's sentences are encoded together in windows of 1024 tokens with stride 768 (`scope = document`). Every token carries its sentence's label in training, and token scores are pooled into one score per sentence.

**Lab** (no training). Each pooling is combined with prior correction strength `tau` in {0, 0.5, 1} and Viterbi transition weight `lam` in {0, 0.5, 1}. Transition probabilities are counted on the training labels.

**Final system.** `decoder_lora` (`r = 8`, `alpha = 16`, learning rate 5e-4), mean pooling, `lam = 0.5`, `tau = 0.5`. Dev macro-F1 0.4599.

| Submission | Official score |
|---|---|
| decoder_lora__mean__lam1.0__tau0.5 | 0.5016 |
| **decoder_lora__mean__lam0.5__tau0.5 (final)** | **0.4922** |
| kd_moe_to_decoder4__mean__lam1.0 | 0.4639 |
| decoder_ia3__last__lam1.0__tau1.0 | 0.4565 |
| swa_bitfit__last__lam1.0__tau1.0 | 0.4491 |
| kd_moe_to_decoder4__last__lam0.5 | 0.4283 |
| greedy__last__lam0.5__tau0.5 | 0.4280 |
| encdec_lorma__mean__lam0.5__tau1.0 | 0.3369 |

## Phase 4: Court judgment prediction

**Task.** Predict the binary outcome of a case from its judgment text.

**Method.** Cases are cut into windows of 1024 tokens. `train_chunks` chooses the training windows, and every window is scored at prediction time. Defaults: 75-minute budget, up to 3 epochs, learning rate 5e-5, batch 8.

**Lab.** Window selection (`all`, `head`, `tail`, `headtail`, `tail2`), token aggregation (`mean`, `last`) and decision threshold from 0.3 to 0.7.

**Final system.** `decoder_partial` (last two layers trainable), all windows, last-token aggregation, threshold 0.4.

| Submission | Official score |
|---|---|
| **decoder_partial__all__last__thr0.4 (final)** | **0.7198** |
| decoder_partial__headtail__last | 0.7126 |
| greedy__tail__last__thr0.4 | 0.6744 |
| greedy__tail__mean__thr0.4 | 0.6734 |
| prefix_full__tail__mean__thr0.6 | 0.6529 |
| moe_full__tail__last__thr0.3 | 0.6473 |
| encdec_lorma__tail2__mean__thr0.4 | 0.5566 |
| swa_bitfit__all__last__thr0.5 | 0.5317 |

## Phase 5: Legal statute identification

**Task.** Predict the set of statutes for each document. The score is macro-F1 over statute ids.

**Method.** One sigmoid output per statute. A document's windows are pooled by `token_pool = mean`, then `doc_pool = max`, with binary cross-entropy. Training samples at most 4 windows per document; up to 16 are scored at prediction time. Defaults: 75-minute budget, up to 3 epochs, learning rate 5e-5.

**Lab.** Window selection, window aggregation (`max`, `mean`, `lse`), decision rule (global threshold, per-statute threshold or top-k) and optional minimum of one label. Per-statute thresholds are fitted on dev and scored by cross-fitting.

**Final system.** Greedy ensemble of `decoder_partial` and `prefix_full`, all windows, `max` aggregation, per-statute thresholds. Cross-fitted validation macro-F1 0.3257.

| Submission | Official score |
|---|---|
| **greedy__all__max__perlabel (final)** | **0.3195** |
| decoder_partial__all__lse__perlabel | 0.3097 |
| decoder_partial__headtail__lse | 0.3079 |
| decoder_partial__headtail__mean | 0.3028 |
| decoder_partial__all__mean | 0.3017 |
| prefix_full__headtail__max | 0.2994 |
| kd_moe_to_decoder4__all__lse | 0.2952 |
| moe_full__all__max__perlabel__min1 | 0.2889 |
| swa_partial__all__max__perlabel | 0.2883 |

## Phase 6: Abstractive summarisation

**Task.** Generate a summary of a long judgment. The score is BERTScore F1.

**Method.** The Phase 1 models are used with their own output layer. Causal and prefix models read `[context] <s> [summary] </s>`, with the loss on summary tokens only. The encoder-decoder reads the context in its encoder. 200 training documents are held out for validation. Defaults: up to 384 summary tokens, 60-minute budget, up to 3 epochs, batch 8, `train_context = head`.

**Lab.** Contexts (`head`, `salient` by tf-idf, `spread`) are combined with decoding variants.

**Final system.** `decoder_lora` (`r = 8`, `alpha = 16`, learning rate 5e-4), `head` context, greedy decoding, `no_repeat_ngram = 3`, 80 to 300 words.

| Submission | Official score |
|---|---|
| **decoder_lora__head__greedy_nr3_w300 (final)** | **0.8379** |
| moe_full__head__greedy_nr3_w300 | 0.8376 |
| decoder_lora__salient__greedy_nr3 | 0.8373 |
| decoder_lora__head__greedy_nr3_w180 | 0.8370 |
| moe_full__spread__greedy_nr3_w300 | 0.8348 |
| moe_full__salient__greedy_rp12_w300 | 0.8342 |
| prefix_lora__salient__greedy_nr3 | 0.8318 |
| kd_moe_to_decoder4__salient__sample | 0.8302 |
| prefix_lora__head__greedy_nr3_w300 | 0.8294 |
| swa_prompt__salient__sample_t07 | 0.8184 |

## Phase 7: Prior case retrieval

**Task.** Return the 7 most relevant prior cases for each query. The score is Micro-F1@7.

**Method.** Documents are cut into windows of 512 tokens. Hidden states are mean-pooled over tokens and normalised. Training uses InfoNCE at temperature 0.05, with BM25 hard negatives (BM25 `k1 = 1.2`, `b = 0.75`) and in-batch negatives. Distillation students also match the teacher's similarity distribution. Up to 12 windows per document are embedded at inference. Defaults: 40-minute budget, up to 4 epochs. A dev re-ranking task on 60 queries chooses each run's checkpoint.

**Lab.** Window aggregations, CSLS hubness correction, BM25 fusion and z-score ensembles of the k best runs.

**Final system.** `ens5`, the mean of z-scored similarities from `decoder_partial`, `swa_prompt`, `kd_moe_to_decoder4`, `moe_full` and `prefix_lorma`. It uses no BM25 fusion. Dev Micro-F1@7 0.3680.

| Submission | Official score |
|---|---|
| **ens5 (final)** | **0.3585** |
| ens2 | 0.3562 |
| ens8 | 0.3556 |
| ens6 | 0.3550 |
| ens3 | 0.3526 |
| ens4 | 0.3520 |
| ens7 | 0.3520 |

## Use of an LLM assistant

An LLM coding assistant was used during this project, including for pipeline, lab and chart code in these notebooks. The notebooks' own introductions say so, and the report's LLM Usage Declaration gives the details.

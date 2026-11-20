# Notes for the report: changes made during the project and why

Newest first. Each entry says what was found, what changed, and the reason.

## 2026-11-20: Third DPO run, with length control and an SFT term

**What was found (second DPO run, official scores, 150 test questions)**

| | 1.7B + SFT, agent | 1.7B + SFT + DPO (2nd run), agent | 4B base, agent | 8B base, agent |
|---|---|---|---|---|
| ROUGE-L | 0.224 | 0.224 | 0.211 | 0.214 |
| BERTScore | 0.173 | 0.237 | 0.248 | 0.243 |
| Faithfulness | 0.702 | 0.852 | 0.726 | 0.722 |
| Loop rate | 40% | 7% | 0% | 0% |
| Words | 90 | 46 | 84 | 92 |

- DPO fixed the loops and raised faithfulness clearly above the 4B and 8B. The faithfulness gain over the 8B was +0.130, with a 95% interval of 0.075 to 0.185.
- But the answers became much shorter: 32 words in rag mode and 46 in agent mode, against 64 for the human summaries.
- A shorter summary has fewer chances to say something unsupported, so part of the faithfulness gain could come from brevity rather than better summaries. ROUGE-1 also dropped (0.335 to 0.300 in rag mode), which shows content was being left out.
- The pairs themselves were only mildly biased: chosen summaries averaged 59 words and rejected 69, and the chosen one was the longer of the two in 43% of pairs. So most of the shortening came from DPO itself. DPO can lower the probability of both summaries in a pair, and ending early becomes an easy way to do that.

**What changed**
- **Length term in the reward:** the reward is now multiplied by the length ratio to the human summary (shorter word count / longer word count). Candidates close to the human length get the most reward, and both very short and very long ones lose some. With the saved candidates this gives 282 pairs, with chosen 62 and rejected 65 words on average, and the chosen one longer in 53% of pairs. Loops still land on the rejected side (42 rejected, 4 chosen).
- **SFT loss on the chosen summary:** this is added to the DPO loss with equal weight (`loss_type=['sigmoid', 'sft']`, known as RPO, Pang et al. 2024). It keeps the probability of the good summaries up, so DPO cannot simply move away from both. The logged training loss is now DPO + SFT, so it starts near 2.9 instead of 0.69.
- **No new candidates:** each saved candidate already has its faithfulness and ROUGE-L scores, so the new rewards are computed from them (`rescore_candidates`). The candidate step (91 minutes) is reused, and only DPO training and the DPO test answers run again.
- **Second run archived:** its adapter, test answers, faithfulness scores, timing rows and result tables are in `old_runs/dpo_second_try/`, so its numbers can still be quoted.
- **Notebook 3:** the significance table now also compares the tuned model with the SFT-only model, to test whether DPO itself made a real difference. The ablation table shows the loop rate.

**Result of the third run (official scores, 150 test questions)**

| | 2nd DPO, rag | 3rd DPO, rag | 2nd DPO, agent | 3rd DPO, agent | 8B base, agent |
|---|---|---|---|---|---|
| ROUGE-1 | 0.300 | 0.365 | 0.314 | 0.350 | 0.353 |
| ROUGE-L | 0.217 | 0.254 | 0.224 | 0.245 | 0.214 |
| BERTScore | 0.258 | 0.260 | 0.237 | 0.242 | 0.243 |
| Faithfulness | 0.869 | 0.879 | 0.852 | 0.858 | 0.722 |
| Hallucination rate | 9% | 6% | 13% | 10% | 21% |
| Loop rate | 1% | 5% | 7% | 9% | 0% |
| Words (human: 64) | 32 | 61 | 46 | 63 | 92 |

- Answer length is back at the human level (61 to 63 words), and faithfulness stayed high. The faithfulness gain was therefore not just from short answers.
- In agent mode the tuned 1.7B is now ahead of every bigger model on both ROUGE-L and faithfulness, and every difference is significant (95% intervals):

| Tuned 1.7B compared with | ROUGE-L difference | Faithfulness difference |
|---|---|---|
| Base 1.7B | +0.043 (0.028 to 0.060) | +0.156 |
| 1.7B with SFT only | +0.021 (0.005 to 0.039) | +0.156 |
| 4B | +0.034 (0.019 to 0.051) | +0.133 |
| 8B | +0.031 (0.015 to 0.048) | +0.137 |
| Ministral 3B | +0.093 | +0.566 |

  The row against SFT only shows that DPO itself made a real difference.
- BERTScore in agent mode is level with the bigger models (0.242 against 0.248 for the 4B and 0.243 for the 8B).
- A few loops remain: 5% of rag answers and 9% of agent answers.
- **Tools:** the tuned model uses a tool on 64% of questions whose starting excerpts missed the evidence, but only on 19% when they had it. The base 1.7B, the 4B and the 8B do not make this distinction.
  - On the questions where it used a tool, its answers are not better than its plain rag answers (ROUGE-L 0.268 against 0.292).
  - Its rag mode also scores slightly higher than its agent mode overall (0.254 against 0.245).
  - So the model learned *when* a tool is needed, but the extra text from tools does not yet make the summaries better.
- **Cost of the final 1.7B pipeline:** about 2.5 hours of training, made up of SFT data 1 min, SFT 33 min, DPO candidates 91 min and DPO training 26 min. All of it fits in 3.4 GB of GPU memory. The tuned 1.7B answers in 4.9 s per question in rag mode, against 6.4 s for the 4B.

## 2026-11-20: Faithfulness not scored in zeroshot mode, scores saved per answer

The final scoring crashed after about 3 hours, on the zeroshot runs:
- In zeroshot mode the model reads the first 6,000 tokens of the transcript, and faithfulness is checked against that text. The longest one is 6,708 tokens for HHEM.
- HHEM needs about 6.6 GB of GPU memory for that length, and its attention memory grows with the square of the length. Using half precision did not reduce it.
- On a GPU smaller than that, the driver borrows regular system memory instead of failing at once. Each score then took about 76 seconds, and in the end the GPU ran out of memory anyway.

What changed:
- Faithfulness is not scored for zeroshot (`FAITH_SKIP_MODES`). Zeroshot is only a baseline, and the faithfulness comparison in the study is about rag and agent mode. Zeroshot still gets ROUGE and BERTScore.
- Every faithfulness score is now saved to `results/faith/` as soon as it is computed. A crash no longer loses finished work, and the significance tests reuse the saved scores instead of scoring every answer again.

## 2026-11-20: DPO redone to fix repetition loops after SFT

**What was found**
- SFT worked well on ROUGE. Rough ROUGE-L in rag mode, same 150 test questions: tuned 1.7B 0.223, base 8B 0.192, base 4B 0.183, base 1.7B 0.183.
- But the SFT model sometimes gets stuck repeating a phrase until it hits the length limit, for example "the team's work, the team's work, ...". This happened in 26% of SFT rag answers (39 of 150 under the loop rule below) and in most general questions in agent mode. Base models never did it (0 of 150 for every base model in rag mode).
- The cause is greedy decoding (always taking the most likely next word), which test runs use so results can be repeated. The Qwen3 model card warns that greedy decoding can lead to endless repetition.
- The first DPO run could not fix it, for two reasons:
  1. Its 4 candidates per question were all sampled at temperature 0.9, and sampled answers almost never loop (24 of 1,200). So DPO never saw the mistake.
  2. The learning rate of 5e-6 was too low for a LoRA adapter. Over 30 steps the loss only went from 0.693 to 0.688, with a reward margin of 0.012 and 69% held-out pair accuracy.

**What changed**
- 1 of the 4 DPO candidates is now written greedily, the same way the model answers at test time (`DPO_GREEDY = 1`). The other 3 are still sampled.
- The DPO reward is multiplied by the share of distinct 4-grams in the summary. Normal text scores about 1, so its reward hardly changes, while a loop scores close to 0 and ends up as the rejected summary. This was needed because a loop repeats words from the transcript, so the faithfulness model alone can still rate it as faithful.
- The DPO learning rate went from 5e-6 to 5e-5. LoRA updates only a small part of the model, so LoRA DPO commonly uses about 10 times the full fine-tuning rate.
- New metric `loop_rate`: the share of answers whose distinct 4-gram share is under 0.8 (`LOOP_BELOW`). The threshold was checked on real outputs. Every base model answer and every human summary scored 0.90 or higher.

**What did not change**
- Every baseline run, the test questions, greedy decoding at test time, and the SFT adapter. A repetition penalty at test time was considered, but it would change every model's answers. The comparison would then only be fair if every baseline was rerun, and the 8B reference runs could not be redone on the local GPU.

**Result of the second DPO run**

| | First DPO (5e-6, sampled only) | Second DPO (5e-5, 1 greedy + 3 sampled) |
|---|---|---|
| Pairs kept (of 300) | 260 | 267 |
| Rejected summaries that loop | 7 | 53 |
| Chosen summaries that loop | 4 | 7 |
| Training loss (start to end) | 0.693 to 0.688 | 0.688 to about 0.5 |
| Held-out reward margin | 0.012 | 0.812 |
| Held-out pair accuracy | 69% | 70% |

The pairs now teach "do not loop" (53 loops on the rejected side), and the model actually changed (the margin is 68 times larger). Candidate time went from 56 to about 91 minutes, because looping greedy answers run to the full 300 tokens. The run was interrupted once at 78 of 300 questions; that session's time (about 23 minutes) is logged in `timings.csv` as an estimate, with tokens taken from the rate of the finished session.

**Story for the report:** SFT teaches the small model the content and style of QMSum summaries, but it introduces greedy repetition loops. DPO with greedy candidates teaches it to avoid them.

**Cost of the first DPO attempt (not counted in the final cost table, kept in `old_runs/dpo_first_try/`):**

| Step | Time | GPU memory |
|---|---|---|
| Candidates | 58 min (including a 1.5-minute crash, see below) | |
| Training | 23 min | 2.8 GB |

## 2026-11-20: Faithfulness scorer ran out of GPU memory

The first DPO candidate step crashed after 2 questions. HHEM scored the 4 candidates in one batch, each against about 2,300 tokens of context, while the 1.7B model was still loaded on the GPU. HHEM's attention memory grows with the square of the input length.

HHEM now scores one pair at a time and clears unused GPU memory first. The scores do not change, only the memory used. The same scorer is used in the final evaluation, where agent contexts can be even longer.

## 2026-11-19: Smaller main model (4B to 1.7B)

- The first plan fine-tuned Qwen3-4B and compared it with Qwen3-8B. The base 4B and 8B were only about 0.01 ROUGE-1 apart, so "gap closed" would mostly measure noise.
- The free cloud GPU quota also ran out.
- So the fine-tuned model became Qwen3-1.7B, with Qwen3-4B (same family) and Ministral 3 3B (another family) as the bigger models, and Qwen3-8B kept as an extra reference. Everything except the 8B runs on a laptop GPU. The finished 4B and 8B runs from the cloud GPU are reused, and the 4B was timed again locally on 20 questions so its speed can be compared.

## 2026-11-19: The small model is taught when to use tools

- In the first runs the base Qwen3-4B called a tool on only 2% of questions (Qwen3-8B on 46%). The base 1.7B also used tools on 2%.
- A model that never calls tools gets nothing from agentic RAG, so 30% of the SFT examples are written in the agent format, built from the gold spans:
  - In half of them the starting excerpts hold the evidence, and the target answers right away.
  - In the other half the gold chunks are removed from the starting excerpts, and the target is a tool call that brings them back, followed by the answer.

## 2026-11-19: Ministral prompts fixed

- Ministral's tokenizer turned the rendered chat text into plain text, so `[INST]` was read as ordinary characters instead of a control token.
- Prompts for that tokenizer are now built directly as token ids. The 40 Ministral answers made before the fix were thrown away and rerun.
- Ministral still scores lower on ROUGE (0.141 rag) because of its style: Markdown bold, openers like "Here's a concise summary", and longer answers (about 127 words against 85 for Qwen).

## 2026-11-19: Cost is measured

Time, peak GPU memory, tokens and estimated FLOPs are logged for every run and training step, since a small model is only worth it if it is also cheaper.

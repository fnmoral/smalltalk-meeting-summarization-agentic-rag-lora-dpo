"""SmallTalk settings: everything you may want to change is in this file.

Edit a value, save, then run the Config cell of the notebook again.
"""

# Where it runs

RUN_ON = 'auto'    # 'local' (6GB laptop), 'colab' (T4), 'server' (remote GPU server) or 'auto' to detect
DOWNLOADS = None   # folder for the dataset and base models, None uses the profile default below

# per machine: only speed and what fits in memory change, never the experiment itself
# 'downloads' paths that are not absolute start at the working folder
PROFILES = {
    # 6GB laptop GPU. downloads stay outside OneDrive so 17GB of models are not synced to the cloud
    'local': {'batch_size': 1, 'big_models': False, 'train': False, 'downloads': '~/smalltalk_downloads'},
    # T4 16GB. the working folder is on Google Drive, so downloads survive a disconnect
    'colab': {'batch_size': 4, 'big_models': True, 'train': True, 'downloads': 'downloads'},
    # remote GPU server, lower the batch size if its GPU is small
    'server': {'batch_size': 8, 'big_models': True, 'train': True, 'downloads': 'downloads'},
}

# Models (key = short name used in file names, value = Hugging Face ID)

SMALL_MODELS = {'qwen3_4b': 'Qwen/Qwen3-4B', 'qwen35_4b': 'Qwen/Qwen3.5-4B'}   # fine-tuned
BIG_MODELS = {'qwen3_8b': 'Qwen/Qwen3-8B', 'qwen35_9b': 'Qwen/Qwen3.5-9B',
              'ministral3_8b': 'mistralai/Ministral-3-8B-Instruct-2512-BF16'}  # comparison only
FAMILY = {'qwen3_4b': 'qwen3_8b', 'qwen35_4b': 'qwen35_9b'}   # same-family big model for each small model
CROSS_FAMILY = 'ministral3_8b'                                # big model from another family
LOAD_4BIT = True   # same 4-bit setting for every model keeps the comparison fair

EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
BGE_QUERY_PROMPT = 'Represent this sentence for searching relevant passages: '
HHEM_MODEL = 'vectara/hallucination_evaluation_model'
NLI_MODEL = 'cross-encoder/nli-deberta-v3-base'   # used only if HHEM fails to load
QMSUM_REPO = 'https://github.com/Yale-LILY/QMSum.git'

# Experiment

SEED = 42
N_TEST = 150             # fixed test subset used by every run, None uses all 281 test questions
K = 8                    # chunks retrieved per question
CHUNK_WORDS = 200        # most spoken words in one chunk
PIECE_WORDS = 150        # a longer speaker turn is cut into pieces of this many words
CHUNK_OVERLAP = 1        # pieces shared between one chunk and the next
ZEROSHOT_TOKENS = 6000   # zeroshot mode reads this many tokens from the start of the transcript
MAX_NEW_TOKENS = 350     # longest answer a model may write
TOP_P = 0.95             # only used when sampling (DPO candidates), test runs are greedy

# Agent

AGENT_MAX_STEPS = 4        # most tool calls per question
TOOL_MAX_CHARS = 3500      # a longer tool result is cut
SEARCH_TOOL_CHUNKS = 4     # new chunks returned by search_transcript
TOPIC_TOOL_CHUNKS = 3      # chunks returned by get_topic_segment
SPEAKER_TOOL_CHUNKS = 6    # chunks searched by get_speaker_turns

# Prompts

SUMMARY_SYSTEM = (
    'You summarize meetings. Answer the question using only the transcript text you are given. '
    'Write a clear summary of 3 to 6 sentences in plain English. '
    'Mention who said what when it matters. Do not add facts that are not in the transcript.')

# keep {tools} and {max_steps}, and keep the words 'Action:', 'Observation' and 'Final Answer:' (the code looks for them)
AGENT_SYSTEM = """You are a meeting assistant. You get a question and some excerpts from the meeting transcript.
If the excerpts are enough to answer the question, answer right away. If something is missing, use a tool to read more of the meeting.

Tools:
{tools}

To use a tool, reply in this exact format:
Thought: what is missing
Action: tool_name[input]

You will then get an Observation with the tool result. You can use at most {max_steps} actions.
When you have enough information, reply with:
Final Answer: a clear summary of 3 to 6 sentences that answers the question, using only facts from the excerpts and observations."""

AGENT_LAST_CALL = 'You have used all your actions. Write the Final Answer now.'

# Fine-tuning (notebook 2)

MODEL_KEY = 'qwen3_4b'   # the small model to fine-tune, then run notebook 2 again with 'qwen35_4b'
RETRAIN = False          # False reuses a saved adapter instead of training again
MAX_LEN = 3072           # most tokens in one training example (prompt + answer)
PROMPT_RESERVE = 300     # tokens kept free for the system prompt, question and chat template
GOLD_CHUNKS = 2          # gold evidence chunks added to each SFT prompt
N_DPO = 300              # train questions kept aside for DPO only
N_VAL = 60               # validation questions used during SFT
SMOKE_STEPS = 5          # training steps in the smoke test

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules='all-linear')

SFT = dict(num_train_epochs=1, learning_rate=2e-4, lr_scheduler_type='cosine', warmup_steps=10,
           per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
           gradient_checkpointing=True, logging_steps=10, save_steps=50, save_total_limit=2)
SFT_EVAL_STEPS = 50

DPO_CANDIDATES = 4          # summaries sampled per question
DPO_TEMPERATURE = 0.9
DPO_MAX_NEW_TOKENS = 300
REWARD_FAITH_WEIGHT = 0.5   # reward = this x faithfulness + the rest x ROUGE-L
DPO_MIN_GAP = 0.1           # best vs worst pairs with a smaller reward gap are dropped

DPO = dict(beta=0.1, num_train_epochs=1, learning_rate=5e-6, lr_scheduler_type='cosine', warmup_steps=5,
           per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
           gradient_checkpointing=True, logging_steps=5, eval_steps=20, save_steps=50, save_total_limit=2)

# Evaluation (notebook 3)

HALLUCINATION_BELOW = 0.5                      # a faithfulness score under this counts as a hallucination
GAP_MODE = 'agent'                             # mode used for the gap-closed table and the main chart
FINAL_RUN = 'runs/qwen3_4b_dpo_agent.jsonl'    # run shown in the error analysis

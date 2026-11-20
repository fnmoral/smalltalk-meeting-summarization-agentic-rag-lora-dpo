"""SmallTalk: shared code for data, RAG, tools, agent, models and metrics.

Imported by the three notebooks so the same pipeline is used everywhere.
All settings and prompts are in config.py.
"""
import gc
import json
import random
import re
import time
from pathlib import Path

import numpy as np

import config as cfg

DATA_DIR = Path('QMSum/data/ALL/jsonl')  # replaced by setup_downloads
MODEL_DIR = None                         # models folder, set by setup_downloads

NOISE = re.compile(r'\{(disfmarker|vocalsound|gap|pause|comment|nonvocalsound)\}')
THINK = re.compile(r'<think>.*?</think>', re.DOTALL)


def set_seed(seed=cfg.SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def free_memory():
    import torch
    gc.collect()
    gc.collect()  # a second pass frees objects released by the first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_gb():
    # GPU memory held by this notebook right now
    import torch
    return torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


TIMING_FIELDS = ['step', 'model', 'seconds', 'peak_gb', 'tokens', 'tflops', 'gpu', 'stopped', 'when']

# estimated FLOPs per token per parameter (Kaplan et al. 2020): a forward pass is 2, SFT with LoRA is 6
# (forward, gradients through the frozen base, and the forward repeated by gradient checkpointing),
# DPO is 8 (the same plus a forward pass of the frozen reference model)
FLOPS_PER_PARAM = {'answer': 2, 'sft': 6, 'dpo': 8}


def flops_per_token(key, kind):
    # 4-bit loading does not change this, the GPU computes in 16-bit
    return FLOPS_PER_PARAM[kind] * cfg.PARAMS_B.get(key, 0) * 1e9


def log_time(step, model, seconds, peak_gb, tokens=0, tflops=0.0, stopped=False, path='results/timings.csv'):
    """Adds one timed step to a csv. Time, memory and compute are reported next to the quality scores."""
    import csv
    import datetime
    new = not Path(path).exists()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        if new:
            w.writeheader()
        w.writerow({'step': step, 'model': model, 'seconds': round(seconds, 1), 'peak_gb': round(peak_gb, 2),
                    'tokens': int(tokens), 'tflops': round(tflops, 1), 'gpu': gpu_name(), 'stopped': stopped,
                    'when': datetime.datetime.now().isoformat(timespec='seconds')})


class Timer:
    """Times a step (for example SFT training) with its peak GPU memory and logs it with log_time.
    Set .tokens and .flops_per_token inside the block to log the estimated compute too.
    A step stopped by an error is still logged and marked as stopped, so resumed steps can be added up."""

    def __init__(self, step, model):
        self.step, self.model = step, model
        self.tokens, self.flops_per_token = 0, 0.0

    def __enter__(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start = time.time()
        return self

    def __exit__(self, kind, err, tb):
        import torch
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        seconds = time.time() - self.start
        tflops = self.tokens * self.flops_per_token / 1e12
        log_time(self.step, self.model, seconds, peak, self.tokens, tflops, stopped=kind is not None)
        print(f'{self.step} ({self.model}): {seconds / 60:.1f} min, peak GPU memory {peak:.1f} GB'
              + (f', {self.tokens:,} tokens, about {tflops:,.0f} TFLOPs' if self.tokens else ''))
        return False


def setup_downloads(path):
    """One folder for everything that is downloaded: the dataset (QMSum/) and the Hugging Face models (models/).
    What is already there is reused, what is missing is downloaded."""
    global DATA_DIR, MODEL_DIR
    import os
    import shutil
    import subprocess
    import sys
    root = Path(path).expanduser().resolve()
    models = root / 'models'
    models.mkdir(parents=True, exist_ok=True)
    # every Hugging Face download (LLMs, embeddings, HHEM, BERTScore) now goes to this folder
    os.environ['HF_HUB_CACHE'] = MODEL_DIR = str(models)
    # if Hugging Face libraries were loaded before this cell they remember the old folder, so update them too
    for name, mod in list(sys.modules.items()):
        if name.split('.')[0] in ('huggingface_hub', 'transformers', 'sentence_transformers'):
            for attr in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE'):
                if isinstance(getattr(mod, attr, None), (str, Path)):
                    setattr(mod, attr, MODEL_DIR)
    if not (root / 'QMSum' / 'data').exists():
        print('Downloading QMSum...')
        subprocess.run(['git', 'clone', '--depth', '1', cfg.QMSUM_REPO, str(root / 'QMSum')], check=True)
    DATA_DIR = root / 'QMSum' / 'data' / 'ALL' / 'jsonl'
    have = sorted(p.name[len('models--'):].replace('--', '/') for p in models.glob('models--*'))
    free = shutil.disk_usage(root).free / 1e9
    print(f'Downloads folder: {root} ({free:.0f} GB free)')
    print('Models already downloaded:', ', '.join(have) if have else 'none yet')
    if free < 25:
        print('Warning: little space left. A model needs about 2 GB per billion parameters (a 4B model about 8 GB).')
    return root


def has_adapter(path):
    # a saved LoRA adapter folder always holds this file
    return (Path(path) / 'adapter_config.json').exists()


def run_profile(run_on='auto'):
    # 'auto' picks colab inside Colab, otherwise server or local by GPU memory
    import sys
    import torch
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    if run_on == 'auto':
        run_on = 'colab' if 'google.colab' in sys.modules else ('server' if gb >= 12 else 'local')
    profile = {'name': run_on, 'gpu_gb': round(gb, 1), **cfg.PROFILES[run_on]}
    print(f"Running on: {run_on} ({profile['gpu_gb']} GB GPU), batch size {profile['batch_size']}, "
          f"models up to {profile['max_params_b']}B, training: {profile['train']}")
    return profile


def fits(key, profile):
    # models bigger than the profile allows are skipped on this machine
    return cfg.PARAMS_B.get(key, 0) <= profile['max_params_b']


# Data

def clean_text(text):
    # remove transcription tags like {disfmarker} and fix spacing before punctuation
    text = NOISE.sub('', text)
    text = re.sub(r'\s+([.,?!])', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_split(split, data_dir=None):
    meetings = []
    with open(Path(data_dir or DATA_DIR) / f'{split}.jsonl', encoding='utf-8') as f:
        for i, line in enumerate(f):
            m = json.loads(line)
            m['id'] = f'{split}_{i}'
            m['turns'] = [{'speaker': t['speaker'], 'text': clean_text(t['content'])}
                          for t in m['meeting_transcripts']]
            m['topics'] = [{'topic': t['topic'],
                            'spans': [(int(a), int(b)) for a, b in t['relevant_text_span']]}
                           for t in m['topic_list']]
            meetings.append(m)
    return meetings


def build_examples(meetings):
    rows = []
    for m in meetings:
        for q in m['general_query_list']:
            rows.append({'meeting_id': m['id'], 'query': q['query'], 'answer': q['answer'],
                         'qtype': 'general', 'spans': []})
        for q in m['specific_query_list']:
            spans = [(int(a), int(b)) for a, b in q['relevant_text_span']]
            rows.append({'meeting_id': m['id'], 'query': q['query'], 'answer': q['answer'],
                         'qtype': 'specific', 'spans': spans})
    return rows


def turns_to_text(turns):
    return '\n'.join(f"{t['speaker']}: {t['text']}" for t in turns if t['text'])


def overlaps(start, end, spans):
    return any(start <= b and a <= end for a, b in spans)


# Chunking and retrieval (LangChain + FAISS)

def split_turns(turns, piece_words=cfg.PIECE_WORDS):
    # long turns (common in committee meetings) are cut into pieces, each piece keeps its turn index
    units = []
    for i, t in enumerate(turns):
        words = t['text'].split()
        for j in range(0, len(words), piece_words):
            units.append((i, f"{t['speaker']}: {' '.join(words[j:j + piece_words])}", len(words[j:j + piece_words])))
    return units


def chunk_meeting(m, max_words=cfg.CHUNK_WORDS, overlap_units=cfg.CHUNK_OVERLAP):
    # chunks hold whole turns where possible and record which turns they cover, to match the gold spans
    from langchain_core.documents import Document
    units, docs, i = split_turns(m['turns']), [], 0
    while i < len(units):
        j, words = i, 0
        while j < len(units) and (words + units[j][2] <= max_words or j == i):
            words += units[j][2]
            j += 1
        group = units[i:j]
        docs.append(Document(page_content='\n'.join(u[1] for u in group),
                             metadata={'meeting_id': m['id'], 'start': group[0][0], 'end': group[-1][0]}))
        if j >= len(units):
            break
        i = max(j - overlap_units, i + 1)
    return docs


class Retriever:
    """One small FAISS index per meeting, built on first use."""

    def __init__(self, embed_model=cfg.EMBED_MODEL, device=None, max_words=cfg.CHUNK_WORDS, embeddings=None):
        if embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            import torch
            device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            embeddings = HuggingFaceEmbeddings(
                model_name=embed_model,
                model_kwargs={'device': device}, cache_folder=MODEL_DIR,
                encode_kwargs={'normalize_embeddings': True},
                query_encode_kwargs={'normalize_embeddings': True, 'prompt': cfg.BGE_QUERY_PROMPT})
        self.emb = embeddings
        self.max_words = max_words
        self.stores, self.chunks = {}, {}

    def index(self, m):
        from langchain_community.vectorstores import FAISS
        if m['id'] not in self.stores:
            docs = chunk_meeting(m, self.max_words)
            self.chunks[m['id']] = docs
            self.stores[m['id']] = FAISS.from_documents(docs, self.emb)
        return self.stores[m['id']]

    def search(self, m, query, k=cfg.K):
        store = self.index(m)
        k = min(k, len(self.chunks[m['id']]))
        return [d for d, _ in store.similarity_search_with_score(query, k=k)]

    def rank_all(self, m, query):
        self.index(m)
        return self.search(m, query, k=len(self.chunks[m['id']]))


def retrieval_metrics(examples, meetings_by_id, retriever, ks=(1, 3, 5, 10)):
    # only specific questions have gold spans
    hits = {k: [] for k in ks}
    recall = {k: [] for k in ks}
    precision = {k: [] for k in ks}
    rr = []
    for ex in examples:
        if ex['qtype'] != 'specific':
            continue
        m = meetings_by_id[ex['meeting_id']]
        ranked = retriever.rank_all(m, ex['query'])
        rel = [overlaps(d.metadata['start'], d.metadata['end'], ex['spans']) for d in ranked]
        n_rel = max(sum(rel), 1)
        first = next((i for i, r in enumerate(rel) if r), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
        for k in ks:
            top = rel[:k]
            hits[k].append(float(any(top)))
            recall[k].append(sum(top) / n_rel)
            precision[k].append(sum(top) / k)
    out = {'n_questions': len(rr), 'MRR': float(np.mean(rr))}
    for k in ks:
        out[f'Hit@{k}'] = float(np.mean(hits[k]))
        out[f'Recall@{k}'] = float(np.mean(recall[k]))
        out[f'Precision@{k}'] = float(np.mean(precision[k]))
    return out


def unique(docs):
    seen, out = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            out.append(d)
    return out


def sort_by_position(docs):
    return unique(sorted(docs, key=lambda d: (d.metadata['start'], d.metadata['end'])))


def format_docs(docs):
    return '\n\n'.join(f"[Turns {d.metadata['start']}-{d.metadata['end']}]\n{d.page_content}" for d in docs)


def ranked_docs(m, ex, retriever, k=cfg.K, add_gold=False):
    # chunks in priority order. general questions: one chunk per labelled topic, specific questions: top-k chunks
    if ex['qtype'] == 'general':
        docs = [retriever.search(m, t['topic'], k=1)[0] for t in m['topics']]
    elif add_gold:
        # for SFT only: the best gold chunks go first, so the evidence is the last thing cut from the prompt
        ranked = retriever.rank_all(m, ex['query'])
        gold = [d for d in ranked if overlaps(d.metadata['start'], d.metadata['end'], ex['spans'])]
        docs = gold[:cfg.GOLD_CHUNKS] + ranked[:k]
    else:
        docs = retriever.search(m, ex['query'], k=k)
    return unique(docs)


def rag_context(m, ex, retriever, k=cfg.K, add_gold=False):
    docs = sort_by_position(ranked_docs(m, ex, retriever, k, add_gold))
    return format_docs(docs), docs


# Prompts (the prompt text is in config.py)

def summary_messages(context, query, label='Transcript excerpts'):
    return [{'role': 'system', 'content': cfg.SUMMARY_SYSTEM},
            {'role': 'user', 'content': f'{label}:\n{context}\n\nQuestion: {query}'}]


# Tools (LangChain) for the agent

def make_tools(m, retriever, max_chars=cfg.TOOL_MAX_CHARS, seen=None):
    from langchain_core.tools import tool

    speakers = sorted({t['speaker'] for t in m['turns']})
    seen = set() if seen is None else seen  # chunks the agent has already read

    def cut(text):
        return text if len(text) <= max_chars else text[:max_chars] + '\n...[cut]'

    @tool
    def search_transcript(query: str) -> str:
        """Search the meeting for parts related to a query that you have not read yet. Input: a short search query."""
        found = retriever.search(m, query, k=3 * cfg.SEARCH_TOOL_CHUNKS)
        new = [d for d in found if d.page_content not in seen][:cfg.SEARCH_TOOL_CHUNKS]
        if not new:
            return 'Nothing new found for that query.'
        seen.update(d.page_content for d in new)
        return cut(format_docs(sort_by_position(new)))

    @tool
    def list_topics(_: str = '') -> str:
        """List the main topics discussed in the meeting and the speakers. Input: nothing."""
        lines = [f"{i + 1}. {t['topic']}" for i, t in enumerate(m['topics'])]
        return 'Topics:\n' + '\n'.join(lines) + '\nSpeakers: ' + ', '.join(speakers)

    @tool
    def get_topic_segment(topic: str) -> str:
        """Read the most important parts of one meeting topic. Input: the topic name or its number from list_topics."""
        topic = topic.strip()
        if topic.isdigit() and 1 <= int(topic) <= len(m['topics']):
            t = m['topics'][int(topic) - 1]
        else:
            t = max(m['topics'], key=lambda t: word_overlap(topic, t['topic']))
        ranked = [d for d in retriever.rank_all(m, t['topic'])
                  if overlaps(d.metadata['start'], d.metadata['end'], t['spans'])]
        docs = sort_by_position(ranked[:cfg.TOPIC_TOOL_CHUNKS])
        body = '\n\n'.join(d.page_content for d in docs)
        return cut(f"Topic: {t['topic']}\n{body}")

    @tool
    def get_speaker_turns(request: str) -> str:
        """Find what one speaker said about something. Input: 'speaker | subject', for example 'Marketing | price'."""
        name, _, about = request.partition('|')
        name = name.strip().lower()
        match = [s for s in speakers if name and (name in s.lower() or s.lower() in name)]
        if not match:
            return 'Unknown speaker. Speakers are: ' + ', '.join(speakers)
        lines = []
        for d in sort_by_position(retriever.search(m, f'{match[0]} {about}', k=cfg.SPEAKER_TOOL_CHUNKS)):
            lines += [ln for ln in d.page_content.split('\n') if ln.startswith(match[0] + ':')]
        return cut('\n'.join(lines) if lines else f'{match[0]} said nothing about that in the closest parts.')

    return {t.name: t for t in [search_transcript, list_topics, get_topic_segment, get_speaker_turns]}


def word_overlap(a, b):
    wa, wb = set(re.findall(r'\w+', a.lower())), set(re.findall(r'\w+', b.lower()))
    return len(wa & wb) / (len(wa) + 1e-9)


# Agent (ReAct loop written by hand so it also works with small models)

# the input runs to the closing bracket at the end of the line, so brackets inside the input are kept
ACTION_RE = re.compile(r'Action\s*:\s*([A-Za-z_]+)\s*(?:\[([^\n]*?)\]|\(([^\n]*?)\))[ \t]*(?=\n|$)')
ACTION_LOOSE = re.compile(r'Action\s*:\s*([A-Za-z_]+)\s*[\[\(](.*?)[\]\)]', re.DOTALL)  # fallback for messy output
FINAL_RE = re.compile(r'Final Answer\s*:\s*(.*)', re.DOTALL | re.IGNORECASE)


def parse_step(text):
    """Return ('action', name, arg), ('final', answer, None) or ('none', text, None)."""
    act, fin = ACTION_RE.search(text) or ACTION_LOOSE.search(text), FINAL_RE.search(text)
    if act and (not fin or act.start() < fin.start()):
        arg = next(g for g in act.groups()[1:] if g is not None)
        return 'action', act.group(1).strip(), arg.strip().strip('\'"')
    if fin:
        return 'final', fin.group(1).strip(), None
    return 'none', text.strip(), None


def agent_messages(ctx, query, tools, max_steps=cfg.AGENT_MAX_STEPS):
    # the same start prompt is used at test time and in the agent SFT examples
    tool_desc = '\n'.join(f'- {name}[input]: {t.description}' for name, t in tools.items())
    return [{'role': 'system', 'content': cfg.AGENT_SYSTEM.format(tools=tool_desc, max_steps=max_steps)},
            {'role': 'user', 'content': f'Transcript excerpts:\n{ctx}\n\nQuestion: {query}'}]


def run_agent(llm, m, ex, retriever, k=cfg.K, max_steps=cfg.AGENT_MAX_STEPS, max_new_tokens=cfg.MAX_NEW_TOKENS):
    # starts from the same chunks as rag mode, then the model chooses to answer or to call tools
    ctx, docs = rag_context(m, ex, retriever, k=k)
    start_hit = any(overlaps(d.metadata['start'], d.metadata['end'], ex['spans']) for d in docs) if ex['spans'] else None
    tools = make_tools(m, retriever, seen={d.page_content for d in docs})
    messages = agent_messages(ctx, ex['query'], tools, max_steps)
    trace, observations, format_errors, answer = [], [], 0, None
    for _ in range(max_steps):
        out = llm.chat(messages, max_new_tokens=max_new_tokens, stop=['Observation:'])
        kind, a, b = parse_step(out)
        if kind != 'action':
            # no 'Final Answer:' label counts as a format error, the text is still used as the answer
            format_errors += kind == 'none'
            answer = a
            break
        if a not in tools:
            obs = f'Unknown tool {a}. Use one of: {", ".join(tools)}'
            format_errors += 1
        else:
            obs = tools[a].invoke(b)
            observations.append(obs)
        trace.append({'tool': a, 'input': b})
        messages += [{'role': 'assistant', 'content': out.split('Observation:')[0].strip()},
                     {'role': 'user', 'content': f'Observation:\n{obs}'}]
    if answer is None:
        messages.append({'role': 'user', 'content': cfg.AGENT_LAST_CALL})
        out = llm.chat(messages, max_new_tokens=max_new_tokens)
        kind, a, _ = parse_step(out)
        answer = a if kind == 'final' else out.strip()
    # context = everything the model read, start_hit = did the starting chunks hold the gold evidence
    return {'pred': answer, 'trace': trace, 'context': '\n\n'.join([ctx] + observations),
            'format_errors': format_errors, 'start_hit': start_hit}


# Models

def compute_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


class LLM:
    """Loads a chat model (optionally 4-bit and/or with a LoRA adapter) and generates text."""

    def __init__(self, model_id, adapter=None, load_4bit=cfg.LOAD_4BIT):
        import torch
        from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.model_id = model_id
        dtype = compute_dtype()
        kwargs = {'device_map': 'auto', 'dtype': dtype, 'cache_dir': MODEL_DIR}
        if load_4bit:
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_DIR)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except (ValueError, KeyError):
            # multimodal checkpoints (for example Ministral 3) need this class, we only use the text part
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Mistral tokenizers must build the ids from the messages, encoding the prompt text would turn [INST] into plain text
        self.ids_template = 'MistralCommon' in type(self.tok).__name__
        self.tokens_in = self.tokens_out = 0   # running totals, used for the FLOPs estimate
        self.device = next(self.model.parameters()).device
        self.torch = torch
        print(f'Loaded {model_id}: {gpu_gb():.1f} GB of GPU memory in use'
              + (' (prompt ids built by the Mistral tokenizer)' if self.ids_template else ''))

    def format(self, messages):
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def encode(self, messages):
        # token ids of the chat prompt, ready for generation
        if not self.ids_template:
            return self.tok(self.format(messages), add_special_tokens=False)['input_ids']
        try:
            out = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        except TypeError:
            out = self.tok.apply_chat_template(messages, tokenize=True)
        ids = out['input_ids'] if hasattr(out, 'keys') else out
        return list(ids[0] if ids and isinstance(ids[0], list) else ids)

    def _tensors(self, ids_list):
        # left padding so all prompts end at the same position
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        n = max(len(x) for x in ids_list)
        ids = self.torch.tensor([[pad] * (n - len(x)) + list(x) for x in ids_list], device=self.device)
        mask = self.torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in ids_list], device=self.device)
        return {'input_ids': ids, 'attention_mask': mask}

    def _count(self, enc, out):
        # prompt tokens (each returned sequence reads the prompt) and new tokens, padding left out
        copies = out.shape[0] // enc['input_ids'].shape[0]
        self.tokens_in += int(enc['attention_mask'].sum()) * copies
        self.tokens_out += int((out[:, enc['input_ids'].shape[1]:] != self.tok.pad_token_id).sum())

    def count_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)['input_ids'])

    def truncate(self, text, max_tokens):
        ids = self.tok(text, add_special_tokens=False)['input_ids']
        return text if len(ids) <= max_tokens else self.tok.decode(ids[:max_tokens])

    def _gen_kwargs(self, max_new_tokens, temperature, stop=None, prompt_len=0):
        kw = {'max_new_tokens': max_new_tokens, 'pad_token_id': self.tok.pad_token_id}
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=cfg.TOP_P)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if stop:
            from transformers import StoppingCriteriaList
            kw['stopping_criteria'] = StoppingCriteriaList([StopOnText(self.tok, stop, prompt_len)])
        return kw

    def chat(self, messages, max_new_tokens=cfg.MAX_NEW_TOKENS, temperature=0.0, stop=None, n=1):
        enc = self._tensors([self.encode(messages)])
        kw = self._gen_kwargs(max_new_tokens, temperature, stop, enc['input_ids'].shape[1])
        with self.torch.no_grad():
            if n == 1 and enc['input_ids'].shape[1] > cfg.PREFILL_CHUNK:
                kw['past_key_values'] = self._prefill(enc['input_ids'])
            out = self.model.generate(**enc, num_return_sequences=n, **kw)
        self._count(enc, out)
        texts = [clean_output(self.tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True))
                 for o in out]
        return texts if n > 1 else texts[0]

    def _prefill(self, ids):
        # reads a long prompt in pieces and keeps the result in the cache. the answer is the same,
        # but attention memory grows with piece x prompt length instead of prompt length squared
        past, last = None, ids.shape[1] - 1
        for i in range(0, last, cfg.PREFILL_CHUNK):
            out = self.model(input_ids=ids[:, i:min(i + cfg.PREFILL_CHUNK, last)], past_key_values=past, use_cache=True)
            past = out.past_key_values
        return past

    def chat_batch(self, batch, max_new_tokens=cfg.MAX_NEW_TOKENS, temperature=0.0):
        if len(batch) == 1:
            return [self.chat(batch[0], max_new_tokens=max_new_tokens, temperature=temperature)]
        enc = self._tensors([self.encode(msgs) for msgs in batch])
        with self.torch.no_grad():
            out = self.model.generate(**enc, **self._gen_kwargs(max_new_tokens, temperature, None))
        self._count(enc, out)
        return [clean_output(self.tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True))
                for o in out]

    def unload(self):
        self.model = None
        free_memory()
        print(f'Unloaded {self.model_id}: {gpu_gb():.1f} GB of GPU memory still in use')

    def memory_check(self, lengths=(2000, 4000, 6000, 8000)):
        """Prints the peak GPU memory for one prompt of each length, read at once and read in pieces."""
        torch = self.torch
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f'GPU total {total:.1f} GB, in use before the test {gpu_gb():.1f} GB')
        token = self.tok(' the', add_special_tokens=False)['input_ids'][0]

        def at_once(ids):
            self.model(input_ids=ids)

        def in_pieces(ids):
            self.model(input_ids=ids[:, -1:], past_key_values=self._prefill(ids))

        for n in lengths:
            ids = torch.full((1, n), token, device=self.device)
            for label, run in [('at once', at_once), ('in pieces', in_pieces)]:
                free_memory()
                torch.cuda.reset_peak_memory_stats()
                try:
                    with torch.no_grad():
                        run(ids)
                    result = f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB'
                except RuntimeError as e:
                    if 'out of memory' not in str(e).lower():
                        raise
                    result = 'OUT OF MEMORY'
                print(f'{n} tokens, {label}: {result}')
        free_memory()
        # the two ways of reading must give the same answer
        text = ' '.join(str(k) for k in range(900))
        msgs = [{'role': 'user', 'content': text + ' What is the last number above? Answer in one short sentence.'}]
        chunk = cfg.PREFILL_CHUNK
        a = self.chat(msgs, max_new_tokens=30)
        cfg.PREFILL_CHUNK = 10**9
        b = self.chat(msgs, max_new_tokens=30)
        cfg.PREFILL_CHUNK = chunk
        print(f'Prompt of {self.count_tokens(self.format(msgs))} tokens, same answer both ways: {a == b}')
        print('  in pieces:', a)
        print('  at once:  ', b)
        free_memory()


class StopOnText:
    """Stops generation once any stop text (for example 'Observation:') appears. Works with any tokenizer."""

    def __init__(self, tok, stops, prompt_len):
        self.tok, self.stops, self.prompt_len = tok, stops, prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        tail = self.tok.decode(input_ids[0, self.prompt_len:][-20:], skip_special_tokens=True)
        done = any(s in tail for s in self.stops)
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


def clean_output(text):
    text = THINK.sub('', text)
    return text.replace('<think>', '').replace('</think>', '').strip()


# Running experiments (results are saved line by line so a Colab disconnect does not lose work)

def done_keys(path):
    if not Path(path).exists():
        return set()
    return {(r['meeting_id'], r['query']) for r in read_jsonl(path)}


def count_todo(path, examples):
    """Questions in examples that have no saved answer in path yet."""
    done = done_keys(path)
    return sum((ex['meeting_id'], ex['query']) not in done for ex in examples)


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    start = ''
    if Path(path).exists() and Path(path).stat().st_size > 0:
        with open(path, 'rb') as f:   # a disconnect can leave the last line unfinished, so start on a new line
            f.seek(-1, 2)
            start = '' if f.read(1) == b'\n' else '\n'
    with open(path, 'a', encoding='utf-8') as f:
        f.write(start + json.dumps(row, ensure_ascii=False) + '\n')


def read_jsonl(path):
    """Skips a line cut short by a disconnect, so that question is simply run again."""
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def run_experiment(llm, mode, examples, meetings_by_id, retriever, out_path, k=cfg.K,
                   zeroshot_tokens=cfg.ZEROSHOT_TOKENS, batch_size=4, max_new_tokens=cfg.MAX_NEW_TOKENS):
    """mode is 'zeroshot', 'rag' or 'agent'. Each answer also saves its time, GPU and token counts,
    and the session's time and peak GPU memory go to results/timings.csv."""
    done = done_keys(out_path)
    todo = [ex for ex in examples if (ex['meeting_id'], ex['query']) not in done]
    print(f'{mode}: {len(done)} done, {len(todo)} to go -> {out_path}')
    if todo:
        start_tokens = llm.tokens_in + llm.tokens_out
        with Timer(f'answer_{mode}', Path(out_path).stem) as timer:
            timer.flops_per_token = flops_per_token(parse_run_name(Path(out_path).stem)[0], 'answer')
            try:
                _answer(llm, mode, todo, meetings_by_id, retriever, out_path, k, zeroshot_tokens, batch_size, max_new_tokens)
            finally:
                timer.tokens = llm.tokens_in + llm.tokens_out - start_tokens


def _answer(llm, mode, todo, meetings_by_id, retriever, out_path, k, zeroshot_tokens, batch_size, max_new_tokens):
    from tqdm.auto import tqdm
    gpu = gpu_name()

    def cost(start, tokens, share=1):
        # time and tokens since start, a batch shares them equally between its questions
        return {'seconds': round((time.time() - start) / share, 2), 'gpu': gpu,
                'tokens_in': round((llm.tokens_in - tokens[0]) / share), 'tokens_out': round((llm.tokens_out - tokens[1]) / share)}

    if mode == 'agent':
        for ex in tqdm(todo):
            m = meetings_by_id[ex['meeting_id']]
            start, tokens = time.time(), (llm.tokens_in, llm.tokens_out)
            res = run_agent(llm, m, ex, retriever, k=k, max_new_tokens=max_new_tokens)
            append_jsonl(out_path, {**ex, **res, 'mode': mode, **cost(start, tokens)})
        return
    if mode == 'zeroshot':
        batch_size = 1  # prompts of 6,000 tokens are too big to batch, attention memory grows with length squared
    for i in tqdm(range(0, len(todo), batch_size)):
        group, batch, contexts = todo[i:i + batch_size], [], []
        for ex in group:
            m = meetings_by_id[ex['meeting_id']]
            if mode == 'zeroshot':
                ctx = llm.truncate(turns_to_text(m['turns']), zeroshot_tokens)
                batch.append(summary_messages(ctx, ex['query'], 'Transcript (may be cut off)'))
            else:
                ctx, _ = rag_context(m, ex, retriever, k=k)
                batch.append(summary_messages(ctx, ex['query']))
            contexts.append(ctx)
        start, tokens = time.time(), (llm.tokens_in, llm.tokens_out)
        try:
            preds = llm.chat_batch(batch, max_new_tokens=max_new_tokens)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            preds = None
        if preds is None:
            # the batch did not fit in GPU memory, so these questions are answered one at a time
            free_memory()
            preds = [llm.chat_batch([msgs], max_new_tokens=max_new_tokens)[0] for msgs in batch]
        c = cost(start, tokens, len(group))
        for ex, ctx, pred in zip(group, contexts, preds):
            append_jsonl(out_path, {**ex, 'pred': pred, 'context': ctx, 'mode': mode, **c})


def has_timing(path):
    rows = read_jsonl(path) if Path(path).exists() else []
    return bool(rows) and 'seconds' in rows[0]


def time_sample(examples, n=cfg.TIME_SAMPLE):
    # evenly spread questions, so the sample keeps the general/specific mix of the test subset
    return examples[::max(len(examples) // n, 1)][:n]


def test_subset(examples, n=None, seed=cfg.SEED):
    """Fixed random subset (same for every run) that keeps the general/specific mix."""
    if n is None or n >= len(examples):
        return examples
    rng = random.Random(seed)
    general = [e for e in examples if e['qtype'] == 'general']
    specific = [e for e in examples if e['qtype'] == 'specific']
    n_gen = round(n * len(general) / len(examples))
    return rng.sample(general, n_gen) + rng.sample(specific, n - n_gen)


def parse_run_name(name):
    # runs are saved as <model_key>_<stage>_<mode>, for example qwen3_17b_dpo_agent
    key, stage, mode = name.rsplit('_', 2)
    return key, stage, mode


# Training data

def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)['input_ids'])


def fit_docs(tok, docs, budget):
    # adds chunks in priority order until the token budget is full, then puts them back in meeting order
    keep, used = [], 0
    for d in docs:
        size = n_tokens(tok, format_docs([d])) + 2
        if used + size <= budget:
            keep.append(d)
            used += size
    return sort_by_position(keep)


def sft_rows(examples, meetings_by_id, retriever, tok, k=cfg.K, max_length=cfg.MAX_LEN):
    """Prompt-completion rows for TRL. Chunks are added by priority (gold evidence first) until the
    token budget is full, so prompt + answer fit in max_length and the evidence is never the part cut."""
    rows = []
    for ex in examples:
        m = meetings_by_id[ex['meeting_id']]
        budget = max_length - n_tokens(tok, ex['answer']) - cfg.PROMPT_RESERVE
        ctx = format_docs(fit_docs(tok, ranked_docs(m, ex, retriever, k=k, add_gold=True), budget))
        rows.append({'prompt': summary_messages(ctx, ex['query']),
                     'completion': [{'role': 'assistant', 'content': ex['answer']}]})
    return rows


def is_gold(d, ex):
    return overlaps(d.metadata['start'], d.metadata['end'], ex['spans'])


def gold_tool_call(m, ex, retriever, start_docs):
    """A tool call whose output holds gold evidence the start excerpts do not have: (tool, input, output) or None.
    Tries search_transcript with the question first, then each labelled topic."""
    gold = [d for d in retriever.rank_all(m, ex['query']) if is_gold(d, ex)]
    calls = [('search_transcript', ex['query'])] + [('get_topic_segment', str(i + 1)) for i in range(len(m['topics']))]
    for name, arg in calls:
        # a fresh tool set each time, because search_transcript remembers what it returned
        obs = make_tools(m, retriever, seen={d.page_content for d in start_docs})[name].invoke(arg)
        if any(d.page_content[:200] in obs for d in gold):   # checks the real output, which may be cut
            return name, arg, obs
    return None


def agent_sft_rows(examples, meetings_by_id, retriever, tok, k=cfg.K, max_length=cfg.MAX_LEN,
                   tool_share=cfg.AGENT_SFT_TOOL_SHARE, seed=cfg.SEED):
    """Agent-format SFT rows built from the gold spans, so the model learns when a tool is needed.
    answer: the start excerpts hold the evidence, so the model answers right away.
    tool: the gold chunks are taken out of the start excerpts, so the model calls the tool that brings
    them back and then answers. This case gives two rows, one for the action and one for the answer."""
    rng = random.Random(seed)
    rows, counts = [], {'answer': 0, 'tool': 0}
    for ex in examples:
        m = meetings_by_id[ex['meeting_id']]
        call = None
        if ex['qtype'] == 'specific' and rng.random() < tool_share:
            ranked = retriever.rank_all(m, ex['query'])
            pool = [d for d in ranked if not is_gold(d, ex)][:k]
            call = gold_tool_call(m, ex, retriever, pool)
        answer = f"Thought: {cfg.SFT_THOUGHT_ANSWER}\nFinal Answer: {ex['answer']}"
        if call is None:
            budget = max_length - n_tokens(tok, answer) - cfg.AGENT_PROMPT_RESERVE
            start = fit_docs(tok, ranked_docs(m, ex, retriever, k=k, add_gold=True), budget)
            tools = make_tools(m, retriever, seen={d.page_content for d in start})
            rows.append({'prompt': agent_messages(format_docs(start), ex['query'], tools),
                         'completion': [{'role': 'assistant', 'content': answer}]})
            counts['answer'] += 1
            continue
        name, arg, obs = call
        action = f'Thought: {cfg.SFT_THOUGHT_TOOL}\nAction: {name}[{arg}]'
        final = f"Final Answer: {ex['answer']}"
        budget = max_length - n_tokens(tok, action + obs + final) - cfg.AGENT_PROMPT_RESERVE
        start = fit_docs(tok, pool, budget)
        prompt = agent_messages(format_docs(start), ex['query'], make_tools(m, retriever))
        rows.append({'prompt': prompt, 'completion': [{'role': 'assistant', 'content': action}]})
        rows.append({'prompt': prompt + [{'role': 'assistant', 'content': action},
                                         {'role': 'user', 'content': f'Observation:\n{obs}'}],
                     'completion': [{'role': 'assistant', 'content': final}]})
        counts['tool'] += 1
    print(f"Agent SFT: {counts['answer']} questions answered right away, {counts['tool']} with a tool call "
          f"({len(rows)} rows)")
    return rows


def train_tokens(dataset, epochs=1):
    """Tokens a trainer reads in its tokenized dataset, for the FLOPs estimate."""
    # SFT rows hold input_ids, DPO rows hold prompt_ids, chosen_ids and rejected_ids (TRL 1.18)
    cols = [c for c in dataset.column_names if c.endswith('_ids')]
    n = sum(len(x) for c in cols for x in dataset[c])
    for c in ('prompt_ids', 'prompt_input_ids'):
        if c in cols:   # DPO reads the prompt twice, once with each answer
            n += sum(len(x) for x in dataset[c])
    return int(n * epochs)


def distinct_share(text, n=4):
    # share of word n-grams that are unique: about 1 for normal text, low for an answer stuck in a loop
    w = text.lower().split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def candidate_reward(candidate, faith_score, rouge_l, answer, weight=cfg.REWARD_FAITH_WEIGHT):
    # (weight * faithfulness + (1 - weight) * ROUGE-L) * distinct 4-gram share * length ratio to the human summary
    # a loop repeats words from the context, so HHEM alone may still call it faithful
    # the length ratio (shorter / longer word count) keeps DPO from learning that shorter is always better
    w, ref = max(len(candidate.split()), 1), max(len(answer.split()), 1)
    return (weight * faith_score + (1 - weight) * rouge_l) * distinct_share(candidate) * min(w, ref) / max(w, ref)


def score_candidates(candidates, context, answer, faith):
    f = faith.score([context] * len(candidates), candidates)
    r = rouge_l_each(candidates, [answer] * len(candidates))
    return [candidate_reward(c, a, b, answer) for c, a, b in zip(candidates, f, r)], f, r


def rescore_candidates(row, answer):
    # rewards of saved candidates with the current formula, from their saved faithfulness and ROUGE-L
    return [candidate_reward(c, a, b, answer) for c, a, b in zip(row['candidates'], row['faith'], row['rougeL'])]


def best_worst_pair(row, min_gap=cfg.DPO_MIN_GAP):
    s = row['scores']
    i, j = int(np.argmax(s)), int(np.argmin(s))
    if s[i] - s[j] < min_gap:
        return None
    return {'prompt': row['prompt'],
            'chosen': [{'role': 'assistant', 'content': row['candidates'][i]}],
            'rejected': [{'role': 'assistant', 'content': row['candidates'][j]}],
            'gap': s[i] - s[j]}


# Metrics

def rouge(preds, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rows = [sc.score(r, p) for p, r in zip(preds, refs)]
    return {k: float(np.mean([row[k].fmeasure for row in rows])) for k in ['rouge1', 'rouge2', 'rougeL']}


def rouge_l_each(preds, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [sc.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]


def bertscore(preds, refs, batch_size=16):
    from bert_score import score
    _, _, f1 = score(preds, refs, lang='en', rescale_with_baseline=True, batch_size=batch_size, verbose=False)
    return float(f1.mean())


class Faithfulness:
    """Checks if a summary is supported by the text the model saw.
    Uses Vectara HHEM-2.1-Open, falls back to an NLI cross-encoder if HHEM fails to load."""

    def __init__(self, prefer='hhem'):
        self.kind = None
        if prefer == 'hhem':
            try:
                from transformers import AutoModelForSequenceClassification
                import torch
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    cfg.HHEM_MODEL, trust_remote_code=True, cache_dir=MODEL_DIR)
                if torch.cuda.is_available():
                    self.model = self.model.to('cuda')
                self.model.eval()
                self.kind = 'hhem'
            except Exception as e:
                print('HHEM did not load, using NLI fallback:', repr(e)[:200])
        if self.kind is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(cfg.NLI_MODEL)
            self.kind = 'nli'
        print('Faithfulness scorer:', self.kind)

    def score(self, premises, hypotheses, batch_size=1):
        # one pair at a time: long contexts in a batch ran out of memory next to a loaded LLM on a small GPU
        if self.kind == 'hhem':
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            out = []
            for i in range(0, len(premises), batch_size):
                pairs = list(zip(premises[i:i + batch_size], hypotheses[i:i + batch_size]))
                with torch.no_grad():
                    out += [float(s) for s in self.model.predict(pairs)]
            return out
        return [self._nli_one(p, h) for p, h in zip(premises, hypotheses)]

    def _nli_one(self, premise, hypothesis, window=300):
        # average over summary sentences of the best entailment score across premise windows
        words = premise.split()
        windows = [' '.join(words[i:i + window]) for i in range(0, max(len(words), 1), window // 2)] or ['']
        sents = [s for s in re.split(r'(?<=[.!?])\s+', hypothesis) if len(s.split()) > 2] or [hypothesis]
        scores = []
        for s in sents:
            logits = self.model.predict([(w, s) for w in windows], apply_softmax=True)
            scores.append(float(np.max(logits[:, 1])))  # label order: contradiction, entailment, neutral
        return float(np.mean(scores))


def faith_scores(path, faith):
    """HHEM score of every answer in a run, in row order. Scores are saved in results/faith/ one line at a time,
    so a crash or a rerun never scores the same answer twice. Modes in FAITH_SKIP_MODES get NaN."""
    rows = read_jsonl(path)
    if rows and rows[0].get('mode') in cfg.FAITH_SKIP_MODES:
        return [float('nan')] * len(rows)
    cache = Path('results/faith') / Path(path).name
    done = {}
    if cache.exists():   # a saved score is only reused if the answer is still the same
        done = {(r['meeting_id'], r['query']): r for r in read_jsonl(cache)}
    out = []
    for r in rows:
        key, pred = (r['meeting_id'], r['query']), r['pred'] or ''
        if key not in done or done[key]['pred'] != pred:
            done[key] = {'meeting_id': key[0], 'query': key[1], 'pred': pred,
                         'faithfulness': faith.score([r['context'] or ''], [pred])[0]}
            append_jsonl(cache, done[key])
        out.append(done[key]['faithfulness'])
    return out


def evaluate_run(path, faith=None, use_bertscore=True):
    rows = read_jsonl(path)
    preds = [r['pred'] or '' for r in rows]
    refs = [r['answer'] for r in rows]
    out = {'run': Path(path).stem, 'n': len(rows), **rouge(preds, refs)}
    if use_bertscore:
        try:
            out['bertscore'] = bertscore(preds, refs)
        except Exception as e:
            print('BERTScore failed, skipping it:', repr(e)[:200])
            out['bertscore'] = float('nan')
    if faith is not None:
        scores = np.array(faith_scores(path, faith))
        out['faithfulness'] = float(np.mean(scores))   # NaN for skipped modes
        out['hallucination_rate'] = float(np.mean(scores < cfg.HALLUCINATION_BELOW)) if not np.isnan(scores).any() else float('nan')
    out['avg_words'] = float(np.mean([len(p.split()) for p in preds]))
    out['loop_rate'] = float(np.mean([distinct_share(p) < cfg.LOOP_BELOW for p in preds]))
    timed = [r['seconds'] for r in rows if 'seconds' in r]
    if timed:   # runs made before timing was added have no times
        out['sec_per_question'] = float(np.mean(timed))
        out['run_minutes'] = float(np.sum(timed)) / 60
        out['gpu'] = rows[-1].get('gpu', '')
    counted = [r['tokens_in'] + r['tokens_out'] for r in rows if 'tokens_in' in r]
    if counted:
        out['tokens_per_question'] = float(np.mean(counted))
        key = parse_run_name(Path(path).stem)[0]
        out['gflops_per_question'] = float(np.mean(counted)) * flops_per_token(key, 'answer') / 1e9
    if rows and 'trace' in rows[0]:
        out['tool_use_rate'] = float(np.mean([len(r['trace']) > 0 for r in rows]))
        out['avg_tool_calls'] = float(np.mean([len(r['trace']) for r in rows]))
        out['format_error_rate'] = float(np.mean([r['format_errors'] > 0 for r in rows]))
    return out


def tool_benefit(agent_path, rag_path):
    """Same model, same questions, same starting chunks: agent answer vs plain rag answer,
    split by whether the agent chose to use a tool."""
    rag = {(r['meeting_id'], r['query']): r for r in read_jsonl(rag_path)}
    pairs = [(a, rag[(a['meeting_id'], a['query'])]) for a in read_jsonl(agent_path)
             if (a['meeting_id'], a['query']) in rag]
    out = []
    for group, used in [('used tools', True), ('no tools', False)]:
        sub = [(a, r) for a, r in pairs if (len(a['trace']) > 0) == used]
        if not sub:
            continue
        refs = [a['answer'] for a, _ in sub]
        with_tools = rouge_l_each([a['pred'] or '' for a, _ in sub], refs)
        without = rouge_l_each([r['pred'] or '' for _, r in sub], refs)
        out.append({'group': group, 'n': len(sub), 'rougeL_agent': float(np.mean(with_tools)),
                    'rougeL_rag': float(np.mean(without)),
                    'difference': float(np.mean(with_tools) - np.mean(without)),
                    'agent_better_share': float(np.mean([x > y for x, y in zip(with_tools, without)]))})
    return out


def tool_decisions(agent_path):
    """Did the agent use tools more often when its starting chunks missed the gold evidence?
    Specific questions only, because only they have gold spans."""
    rows = [r for r in read_jsonl(agent_path) if r.get('start_hit') is not None]
    out = []
    for group, hit in [('start had the evidence', True), ('start missed the evidence', False)]:
        sub = [r for r in rows if r['start_hit'] == hit]
        if sub:
            out.append({'group': group, 'n': len(sub),
                        'tool_use_rate': float(np.mean([len(r['trace']) > 0 for r in sub]))})
    return out


def paired_bootstrap(a, b, n=2000, seed=cfg.SEED):
    """Mean of b - a over the same questions with a 95% interval. If the interval does not include 0,
    the difference is unlikely to be luck of which questions were in the test set."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {'difference': float(d.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
            'real_difference': bool(lo > 0 or hi < 0), 'b_better_share': float(np.mean(d > 0))}


def compare_runs(path_a, path_b, faith=None):
    """Per-question comparison of two runs on the questions both answered (b - a)."""
    a = {(r['meeting_id'], r['query']): r for r in read_jsonl(path_a)}
    b = {(r['meeting_id'], r['query']): r for r in read_jsonl(path_b)}
    keys = [key for key in a if key in b]
    refs = [a[key]['answer'] for key in keys]
    pa, pb = [a[key]['pred'] or '' for key in keys], [b[key]['pred'] or '' for key in keys]
    out = [{'metric': 'rougeL', 'n': len(keys), **paired_bootstrap(rouge_l_each(pa, refs), rouge_l_each(pb, refs))}]
    if faith is not None:   # saved scores from faith_scores, matched by question
        sa = dict(zip([(r['meeting_id'], r['query']) for r in read_jsonl(path_a)], faith_scores(path_a, faith)))
        sb = dict(zip([(r['meeting_id'], r['query']) for r in read_jsonl(path_b)], faith_scores(path_b, faith)))
        fa, fb = [sa[key] for key in keys], [sb[key] for key in keys]
        if not np.isnan(fa + fb).any():
            out.append({'metric': 'faithfulness', 'n': len(keys), **paired_bootstrap(fa, fb)})
    return out


def gap_closed(small_base, small_tuned, big):
    """Share of the gap between the small base model and the big model closed by fine-tuning."""
    gap = big - small_base
    return float('nan') if abs(gap) < 1e-9 else (small_tuned - small_base) / gap

"""SmallTalk: shared code for data, RAG, tools, agent, models and metrics.

Imported by the three notebooks so the same pipeline is used everywhere.
All settings and prompts are in config.py.
"""
import gc
import json
import random
import re
from pathlib import Path

import numpy as np

import config as cfg

DATA_DIR = Path('QMSum/data/ALL/jsonl')  # replaced by setup_downloads

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_downloads(path):
    """One folder for everything that is downloaded: the dataset (QMSum/) and the Hugging Face models (models/).
    What is already there is reused, what is missing is downloaded."""
    global DATA_DIR
    import os
    import shutil
    import subprocess
    import sys
    root = Path(path).expanduser().resolve()
    models = root / 'models'
    models.mkdir(parents=True, exist_ok=True)
    # every Hugging Face download (LLMs, embeddings, HHEM, BERTScore) now goes to this folder
    os.environ['HF_HUB_CACHE'] = str(models)
    hub = sys.modules.get('huggingface_hub.constants')
    if hub is not None and Path(hub.HF_HUB_CACHE).resolve() != models:
        print('Warning: Hugging Face was loaded before this cell, so models still go to', hub.HF_HUB_CACHE)
        print('Restart the kernel and run the notebook from the top to fix this.')
    if not (root / 'QMSum' / 'data').exists():
        print('Downloading QMSum...')
        subprocess.run(['git', 'clone', '--depth', '1', cfg.QMSUM_REPO, str(root / 'QMSum')], check=True)
    DATA_DIR = root / 'QMSum' / 'data' / 'ALL' / 'jsonl'
    have = sorted(p.name[len('models--'):].replace('--', '/') for p in models.glob('models--*'))
    free = shutil.disk_usage(root).free / 1e9
    print(f'Downloads folder: {root} ({free:.0f} GB free)')
    print('Models already downloaded:', ', '.join(have) if have else 'none yet')
    if free < 25:
        print('Warning: little space left. A 4B model needs about 9 GB and an 8B/9B model about 18 GB.')
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
          f"big models: {profile['big_models']}, training: {profile['train']}")
    return profile


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
                model_kwargs={'device': device},
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


def run_agent(llm, m, ex, retriever, k=cfg.K, max_steps=cfg.AGENT_MAX_STEPS, max_new_tokens=cfg.MAX_NEW_TOKENS):
    # starts from the same chunks as rag mode, then the model chooses to answer or to call tools
    ctx, docs = rag_context(m, ex, retriever, k=k)
    start_hit = any(overlaps(d.metadata['start'], d.metadata['end'], ex['spans']) for d in docs) if ex['spans'] else None
    tools = make_tools(m, retriever, seen={d.page_content for d in docs})
    tool_desc = '\n'.join(f'- {name}[input]: {t.description}' for name, t in tools.items())
    messages = [{'role': 'system', 'content': cfg.AGENT_SYSTEM.format(tools=tool_desc, max_steps=max_steps)},
                {'role': 'user', 'content': f"Transcript excerpts:\n{ctx}\n\nQuestion: {ex['query']}"}]
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
        kwargs = {'device_map': 'auto', 'dtype': dtype}
        if load_4bit:
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(model_id)
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
        self.device = next(self.model.parameters()).device
        self.torch = torch

    def format(self, messages):
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        prompt = self.format(messages)
        enc = self.tok(prompt, return_tensors='pt', add_special_tokens=False).to(self.device)
        kw = self._gen_kwargs(max_new_tokens, temperature, stop, enc['input_ids'].shape[1])
        with self.torch.no_grad():
            out = self.model.generate(**enc, num_return_sequences=n, **kw)
        texts = [clean_output(self.tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True))
                 for o in out]
        return texts if n > 1 else texts[0]

    def chat_batch(self, batch, max_new_tokens=cfg.MAX_NEW_TOKENS, temperature=0.0):
        # left padding so all prompts end at the same position
        self.tok.padding_side = 'left'
        prompts = [self.format(msgs) for msgs in batch]
        enc = self.tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, **self._gen_kwargs(max_new_tokens, temperature, None))
        return [clean_output(self.tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True))
                for o in out]

    def unload(self):
        del self.model
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
    with open(path, encoding='utf-8') as f:
        return {(r['meeting_id'], r['query']) for r in map(json.loads, f)}


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def run_experiment(llm, mode, examples, meetings_by_id, retriever, out_path, k=cfg.K,
                   zeroshot_tokens=cfg.ZEROSHOT_TOKENS, batch_size=4, max_new_tokens=cfg.MAX_NEW_TOKENS):
    """mode is 'zeroshot', 'rag' or 'agent'."""
    from tqdm.auto import tqdm
    done = done_keys(out_path)
    todo = [ex for ex in examples if (ex['meeting_id'], ex['query']) not in done]
    print(f'{mode}: {len(done)} done, {len(todo)} to go -> {out_path}')
    if mode == 'agent':
        for ex in tqdm(todo):
            m = meetings_by_id[ex['meeting_id']]
            res = run_agent(llm, m, ex, retriever, k=k, max_new_tokens=max_new_tokens)
            append_jsonl(out_path, {**ex, **res, 'mode': mode})
        return
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
        preds = llm.chat_batch(batch, max_new_tokens=max_new_tokens)
        for ex, ctx, pred in zip(group, contexts, preds):
            append_jsonl(out_path, {**ex, 'pred': pred, 'context': ctx, 'mode': mode})


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
    # runs are saved as <model_key>_<stage>_<mode>, for example qwen3_4b_dpo_agent
    key, stage, mode = name.rsplit('_', 2)
    return key, stage, mode


# Training data

def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)['input_ids'])


def sft_rows(examples, meetings_by_id, retriever, tok, k=cfg.K, max_length=cfg.MAX_LEN):
    """Prompt-completion rows for TRL. Chunks are added by priority (gold evidence first) until the
    token budget is full, so prompt + answer fit in max_length and the evidence is never the part cut."""
    rows = []
    for ex in examples:
        m = meetings_by_id[ex['meeting_id']]
        budget = max_length - n_tokens(tok, ex['answer']) - cfg.PROMPT_RESERVE
        keep, used = [], 0
        for d in ranked_docs(m, ex, retriever, k=k, add_gold=True):
            size = n_tokens(tok, format_docs([d])) + 2
            if used + size <= budget:
                keep.append(d)
                used += size
        ctx = format_docs(sort_by_position(keep))
        rows.append({'prompt': summary_messages(ctx, ex['query']),
                     'completion': [{'role': 'assistant', 'content': ex['answer']}]})
    return rows


def score_candidates(candidates, context, answer, faith, weight=cfg.REWARD_FAITH_WEIGHT):
    # reward = weight * faithfulness to the context + (1 - weight) * ROUGE-L against the human summary
    f = faith.score([context] * len(candidates), candidates)
    r = rouge_l_each(candidates, [answer] * len(candidates))
    return [weight * a + (1 - weight) * b for a, b in zip(f, r)], f, r


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
                self.model = AutoModelForSequenceClassification.from_pretrained(cfg.HHEM_MODEL, trust_remote_code=True)
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

    def score(self, premises, hypotheses, batch_size=4):
        if self.kind == 'hhem':
            import torch
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
        scores = faith.score([r['context'] or '' for r in rows], preds)
        out['faithfulness'] = float(np.mean(scores))
        out['hallucination_rate'] = float(np.mean([s < cfg.HALLUCINATION_BELOW for s in scores]))
    out['avg_words'] = float(np.mean([len(p.split()) for p in preds]))
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


def gap_closed(small_base, small_tuned, big):
    """Share of the gap between the small base model and the big model closed by fine-tuning."""
    gap = big - small_base
    return float('nan') if abs(gap) < 1e-9 else (small_tuned - small_base) / gap

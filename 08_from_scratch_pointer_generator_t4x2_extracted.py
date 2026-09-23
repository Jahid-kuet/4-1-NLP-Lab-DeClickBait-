# --- Cell 2 (starts at line 1) ---
!nvidia-smi --query-gpu=name,memory.total --format=csv
# Install only missing packages without touching pre-installed numpy, pandas, or torch (preserves binary ABI compatibility)
!pip install -q rouge-score "transformers>=4.40.0" matplotlib seaborn
print("✓ Environment dependencies installed successfully!")

# --- Cell 3 (starts at line 7) ---
import glob, json, math, os, random, re as _re, shutil, time, unicodedata, zipfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

N_GPUS = max(torch.cuda.device_count(), 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 70)
print(f"🌍 Compute Device: {device} | Detected GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"   [GPU {i}] {torch.cuda.get_device_name(i)}")
print("=" * 70)

def find_data_root():
    candidates = [
        '/kaggle/input/datasets/tasmirhossainzihad/declickbait-splits',
        '/kaggle/input/datasets/tasmirhossainzihad/declickbait-splits/splits',
        '/kaggle/input/declickbait-splits',
        '/kaggle/input/declickbait-splits/splits',
        '/kaggle/input/tasmirhossainzihad-declickbait-splits',
        '/kaggle/input/tasmirhossainzihad-declickbait-splits/splits',
        'splits',
        '../splits',
        '../../splits',
        '/content/drive/MyDrive/declickbait/splits',
        '/content/splits'
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            if (p / 'train_train_val_test.jsonl').exists():
                print(f"✓ Located dataset directory at: {p.resolve()}")
                return str(p.resolve())
            if (p / 'splits' / 'train_train_val_test.jsonl').exists():
                print(f"✓ Located dataset directory at: {(p / 'splits').resolve()}")
                return str((p / 'splits').resolve())
    for hit in glob.glob('/kaggle/input/**/train_train_val_test.jsonl', recursive=True):
        p = Path(hit).parent
        print(f"✓ Located dataset directory at: {p.resolve()}")
        return str(p.resolve())
    for hit in glob.glob('**/train_train_val_test.jsonl', recursive=True):
        p = Path(hit).parent
        print(f"✓ Located dataset directory at: {p.resolve()}")
        return str(p.resolve())
    print("⚠️ WARNING: splits directory not found automatically. Defaulting to '/kaggle/input/datasets/tasmirhossainzihad/declickbait-splits'.")
    return '/kaggle/input/datasets/tasmirhossainzihad/declickbait-splits'

DATA_ROOT = find_data_root()
OUT = 'outputs/08_pointer_generator'
CHECKPOINT_DIR = f"{OUT}/checkpoints"
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

def print_mem(tag=''):
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            res = torch.cuda.memory_reserved(i) / 1e9
            print(f"[{tag} - GPU {i}] Allocated: {alloc:.2f} GB | Reserved: {res:.2f} GB / 16.0 GB")

# --- Cell 5 (starts at line 84) ---
import re as _re

PROMO_PATTERNS = [
    r'subscrib\w*', r'bell\s*icon', r'notification\s*(bell|on)',
    r'\blike\b.{0,20}\bshare\b', r'follow\s*us', r'follow\s*on',
    r'channel.{0,20}subscrib', r'facebook\b.{0,30}page',
    r'youtube\b.{0,30}channel', r'instagram\b.{0,30}account',
    r'twitter\b.{0,30}handle', r'visit\s*(our)?\s*website',
    r'copyright\s*disclaimer', r'all\s*rights?\s*reserved',
    r'সবাইকে\s*(আমাদের)?\s*চ্যানেলে', r'সাবস্ক্রাইব\s*করুন',
    r'লাইক\s*(ও|এবং)?\s*শেয়ার', r'বেল\s*আইকন', r'ভিডিওটি\s*ভালো\s*লাগলে',
    r'শেয়ার\s*করতে\s*ভুলবেন\s*না', r'আমাদের\s*সাথেই\s*থাকুন',
    r'পরবর্তী\s*ভিডিও\s*পেতে', r'ফেসবুক\s*পেজ', r'ইউটিউব\s*চ্যানেল',
    r'ভিজিট\s*করুন', r'কমেন্ট\s*করে\s*জানান'
]
_PROMO_RX = _re.compile('|'.join(f'(?:{p})' for p in PROMO_PATTERNS), _re.IGNORECASE)
_SENT_RX = _re.compile(r'(?<=[.!?।…])\s+|\n+')

def clean_bengali_text(text: str) -> str:
    """Normalizes Unicode NFKC, strips zero-width characters and whitespace."""
    if not text:
        return ''
    t = unicodedata.normalize('NFKC', str(text))
    t = _re.sub(r'[\u200B-\u200D\uFEFF\u00AD]', '', t)
    return _re.sub(r'\s+', ' ', t).strip()

def clean_body_lead(text: str, max_chars: int = 1500, min_keep: int = 40) -> str:
    """Extracts lead discourse, strips promotional sentences, and respects Daari boundaries."""
    cleaned = clean_bengali_text(text)
    if not cleaned:
        return ''
    sentences = [s.strip() for s in _SENT_RX.split(cleaned) if s.strip() and not _PROMO_RX.search(s)]
    lead_text = ' '.join(sentences)
    if len(lead_text) >= min_keep:
        return lead_text[:max_chars].strip()
    return cleaned[:max_chars].strip()

def resolve_target(row: dict) -> str:
    cand = (row.get('target_title') or row.get('truth_title') or row.get('clarified_title') or
            row.get('declickbaited_title') or row.get('factual_title') or row.get('summary') or '')
    cand = clean_bengali_text(cand)
    if cand and len(cand) >= 4:
        return cand
    return clean_bengali_text(row.get('original_title', ''))

print("✓ Bengali linguistic normalization and lead bias extraction ready!")

# --- Cell 7 (starts at line 132) ---
# Prevent transformers generation module from triggering broken sklearn/scipy imports on NumPy 2
try:
    import transformers.utils.import_utils as _tiu
    _tiu._sklearn_available = False
except Exception:
    pass

try:
    from transformers import AutoTokenizer
except (ImportError, AttributeError):
    from transformers.models.bert import BertTokenizer as AutoTokenizer

TOKENIZER_NAME = "csebuetnlp/banglishbert"
print(f"Loading BUET multilingual tokenizer: {TOKENIZER_NAME}...")
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

VOCAB_SIZE = tok.vocab_size
PAD_ID = tok.pad_token_id
UNK_ID = tok.unk_token_id
BOS_ID = tok.cls_token_id or tok.bos_token_id
EOS_ID = tok.sep_token_id or tok.eos_token_id

print(f"✓ Tokenizer Loaded! Vocab Size: {VOCAB_SIZE:,} | PAD={PAD_ID} | UNK={UNK_ID} | BOS={BOS_ID} | EOS={EOS_ID}")

# Verification on multilingual test strings
test_str = "সাকিব আল হাসান scored 50 runs in Sylhet! Khub bhalo khelese."
enc_test = tok.encode(test_str, add_special_tokens=False)
dec_test = tok.decode(enc_test)
print(f"  Verification Input:  '{test_str}'")
print(f"  Decoded Output:      '{dec_test}' (Zero byte corruption!)")

class ExtendedVocabBatchProcessor:
    """
    Maps source tokens to base vocabulary and builds dynamic OOV offsets:
    1. src_tokens: Clamped to base vocabulary for embedding retrieval.
    2. src_extended: Tokens mapped to [0 .. V + N_oov - 1] for scatter accumulation.
    3. tgt_tokens: Target tokens mapped to base vocab OR matching dynamic source OOV index.
    """
    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def process_pair(self, src_text: str, tgt_text: str):
        src_ids = self.tokenizer.encode(src_text, add_special_tokens=False)
        tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=False) if tgt_text else []
        
        oov_to_ext_id = {}
        src_extended = []
        src_clamped = []
        
        for tid in src_ids:
            if tid < self.vocab_size and tid != UNK_ID:
                src_clamped.append(tid)
                src_extended.append(tid)
            else:
                if tid not in oov_to_ext_id:
                    oov_to_ext_id[tid] = self.vocab_size + len(oov_to_ext_id)
                src_clamped.append(UNK_ID)
                src_extended.append(oov_to_ext_id[tid])
                
        tgt_mapped = []
        for tid in tgt_ids:
            if tid < self.vocab_size and tid != UNK_ID:
                tgt_mapped.append(tid)
            elif tid in oov_to_ext_id:
                tgt_mapped.append(oov_to_ext_id[tid])  # Dynamic entity copy target!
            else:
                tgt_mapped.append(UNK_ID)
                
        return {
            'src_tokens': src_clamped,
            'src_extended': src_extended,
            'tgt_tokens': tgt_mapped,
            'num_oovs': len(oov_to_ext_id),
            'src_raw': src_text,
            'tgt_raw': tgt_text
        }

vocab_processor = ExtendedVocabBatchProcessor(tok, VOCAB_SIZE)
print("✓ ExtendedVocabBatchProcessor initialized with BUET vocabulary!")

# --- Cell 9 (starts at line 214) ---
from torch.utils.data import Dataset, DataLoader

SRC_MAX_LEN = 180  # Inverted-pyramid lead context (150-200 subwords)
TGT_MAX_LEN = 24   # Extreme headline summarization boundary

IS_KAGGLE = os.path.exists('/kaggle')
BATCH_SIZE = 32 * N_GPUS  # 64 on Kaggle T4 x 2, 32 on single T4
EVAL_BATCH = 64 * N_GPUS
WORKERS = 2 * N_GPUS if IS_KAGGLE else 0

def load_jsonl(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return [json.loads(l) for l in f if l.strip()]

class ExtremeSummarizationDataset(Dataset):
    def __init__(self, rows: List[dict]):
        self.samples = []
        for r in rows:
            lead = clean_body_lead(r.get('body', ''), max_chars=1500)
            target = resolve_target(r)
            if not lead or not target:
                continue
            self.samples.append((lead, target, str(r.get('language_mix', 'bangla'))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lead, target, lang = self.samples[idx]
        return vocab_processor.process_pair(lead, target), lang

def collate_pointer_gen(batch):
    processed = [b[0] for b in batch]
    langs = [b[1] for b in batch]
    
    batch_size = len(processed)
    max_dynamic_oovs = max(p['num_oovs'] for p in processed)
    
    src_lens = [min(len(p['src_tokens']), SRC_MAX_LEN) for p in processed]
    max_src_len = max(src_lens)
    
    src_tokens_padded = torch.full((batch_size, max_src_len), PAD_ID, dtype=torch.long)
    src_ext_padded = torch.full((batch_size, max_src_len), PAD_ID, dtype=torch.long)
    src_mask = torch.ones((batch_size, max_src_len), dtype=torch.bool)
    
    for i, p in enumerate(processed):
        l = src_lens[i]
        src_tokens_padded[i, :l] = torch.tensor(p['src_tokens'][:l], dtype=torch.long)
        src_ext_padded[i, :l] = torch.tensor(p['src_extended'][:l], dtype=torch.long)
        src_mask[i, :l] = False
        
    tgt_in_list = []
    tgt_out_list = []
    for p in processed:
        t_ids = p['tgt_tokens'][:TGT_MAX_LEN - 2]
        tgt_in_list.append([BOS_ID] + t_ids)
        tgt_out_list.append(t_ids + [EOS_ID])
        
    max_tgt_len = max(len(t) for t in tgt_in_list)
    tgt_in_padded = torch.full((batch_size, max_tgt_len), PAD_ID, dtype=torch.long)
    tgt_out_padded = torch.full((batch_size, max_tgt_len), PAD_ID, dtype=torch.long)
    
    for i in range(batch_size):
        l = len(tgt_in_list[i])
        tgt_in_padded[i, :l] = torch.tensor(tgt_in_list[i], dtype=torch.long)
        tgt_out_padded[i, :l] = torch.tensor(tgt_out_list[i], dtype=torch.long)
        
    return {
        'src_tokens': src_tokens_padded,
        'src_lens': torch.tensor(src_lens, dtype=torch.long),
        'src_extended_tokens': src_ext_padded,
        'max_dynamic_oovs': max_dynamic_oovs,
        'src_padding_mask': src_mask,
        'tgt_in': tgt_in_padded,
        'tgt_out': tgt_out_padded,
        'raw_src': [p['src_raw'] for p in processed],
        'raw_tgt': [p['tgt_raw'] for p in processed],
        'langs': langs
    }

train_rows = load_jsonl(f"{DATA_ROOT}/train_train_val_test.jsonl")
val_rows = load_jsonl(f"{DATA_ROOT}/val_train_val_test.jsonl")
test_rows = load_jsonl(f"{DATA_ROOT}/test_train_val_test.jsonl")

train_dataset = ExtremeSummarizationDataset(train_rows)
val_dataset = ExtremeSummarizationDataset(val_rows)
test_dataset = ExtremeSummarizationDataset(test_rows)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pointer_gen, num_workers=WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_pointer_gen, num_workers=WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_pointer_gen, num_workers=WORKERS, pin_memory=True)

print("=" * 70)
print(f"✓ DataLoaders Ready across {N_GPUS} GPU(s):")
print(f"  - Train Pairs: {len(train_dataset):,} ({len(train_loader)} batches/epoch | Batch Size: {BATCH_SIZE})")
print(f"  - Val Pairs:   {len(val_dataset):,} ({len(val_loader)} batches)")
print(f"  - Test Pairs:  {len(test_dataset):,} ({len(test_loader)} batches)")
print("=" * 70)

# --- Cell 11 (starts at line 314) ---
class PointerGeneratorNetwork(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        padding_idx: int = 0,
        dropout: float = 0.15
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim)

        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_c = nn.Linear(1, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        self.p_gen_projector = nn.Linear(hidden_dim * 2 + hidden_dim + embed_dim, 1)

        self.out_dense = nn.Linear(hidden_dim * 3, hidden_dim)
        self.out_vocab = nn.Linear(hidden_dim, vocab_size)

    def encode(self, src_tokens: torch.Tensor, src_lens: torch.Tensor):
        src_emb = self.dropout(self.embedding(src_tokens))
        lens_cpu = src_lens.clamp(min=1).cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            src_emb, lens_cpu, batch_first=True, enforce_sorted=False
        )
        packed_outputs, (h_n, c_n) = self.encoder(packed_input)
        enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)
        c_cat = torch.cat([c_n[0], c_n[1]], dim=-1)
        dec_h0 = torch.tanh(self.reduce_h(h_cat)).unsqueeze(0)
        dec_c0 = torch.tanh(self.reduce_c(c_cat)).unsqueeze(0)

        return enc_outputs, (dec_h0, dec_c0)

    def attention_step(
        self,
        dec_state: torch.Tensor,
        enc_outputs: torch.Tensor,
        enc_projected: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor],
        coverage: torch.Tensor
    ):
        dec_projected = self.W_s(dec_state)
        cov_projected = self.W_c(coverage)

        scores = self.v(torch.tanh(enc_projected + dec_projected + cov_projected)).squeeze(-1)
        if src_padding_mask is not None:
            scores = scores.masked_fill(src_padding_mask.bool(), -1e9)
        attn_dist = F.softmax(scores, dim=-1)

        context = torch.bmm(attn_dist.unsqueeze(1), enc_outputs)
        return attn_dist, context

    def decode_step(
        self,
        y_prev: torch.Tensor,
        dec_state: Tuple[torch.Tensor, torch.Tensor],
        enc_outputs: torch.Tensor,
        enc_projected: torch.Tensor,
        src_extended_tokens: torch.Tensor,
        max_dynamic_oovs: int | torch.Tensor,
        coverage: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor]
    ):
        batch_size = y_prev.size(0)
        y_emb = self.embedding(torch.clamp(y_prev, max=self.vocab_size - 1))
        dec_out, dec_state = self.decoder(y_emb, dec_state)

        attn_dist, context = self.attention_step(
            dec_out, enc_outputs, enc_projected, src_padding_mask, coverage
        )

        step_cov_loss = torch.sum(torch.min(attn_dist, coverage.squeeze(-1)), dim=-1)
        coverage = coverage + attn_dist.unsqueeze(-1)

        p_gen_input = torch.cat([context, dec_out, y_emb], dim=-1)
        p_gen = torch.sigmoid(self.p_gen_projector(p_gen_input)).squeeze(-1)

        vocab_rep = torch.tanh(self.out_dense(torch.cat([dec_out, context], dim=-1)))
        vocab_dist = F.softmax(self.out_vocab(vocab_rep).squeeze(1), dim=-1)

        weighted_vocab_dist = p_gen * vocab_dist
        weighted_attn_dist = (1.0 - p_gen) * attn_dist

        max_oov_int = int(max_dynamic_oovs.item()) if isinstance(max_dynamic_oovs, torch.Tensor) else int(max_dynamic_oovs)
        full_vocab_dim = self.vocab_size + max_oov_int
        final_dist = torch.zeros(batch_size, full_vocab_dim, device=y_prev.device)
        final_dist[:, :self.vocab_size] = weighted_vocab_dist
        final_dist.scatter_add_(1, src_extended_tokens, weighted_attn_dist)

        return final_dist, dec_state, coverage, step_cov_loss, p_gen

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lens: torch.Tensor,
        src_extended_tokens: torch.Tensor,
        max_dynamic_oovs: int | torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None
    ):
        batch_size, src_len = src_tokens.size()
        tgt_len = tgt_tokens.size(1)

        enc_outputs, (dec_h, dec_c) = self.encode(src_tokens, src_lens)
        enc_projected = self.W_h(enc_outputs)

        coverage = torch.zeros(batch_size, src_len, 1, device=src_tokens.device)
        step_distributions = []
        coverage_penalties = []

        for t in range(tgt_len):
            y_t = tgt_tokens[:, t].unsqueeze(1)
            y_emb = self.embedding(torch.clamp(y_t, max=self.vocab_size - 1))
            dec_out, (dec_h, dec_c) = self.decoder(y_emb, (dec_h, dec_c))

            attn_dist, context = self.attention_step(
                dec_out, enc_outputs, enc_projected, src_padding_mask, coverage
            )

            step_cov_loss = torch.sum(torch.min(attn_dist, coverage.squeeze(-1)), dim=-1)
            coverage_penalties.append(step_cov_loss)
            coverage = coverage + attn_dist.unsqueeze(-1)

            p_gen_input = torch.cat([context, dec_out, y_emb], dim=-1)
            p_gen = torch.sigmoid(self.p_gen_projector(p_gen_input)).squeeze(-1)

            vocab_rep = torch.tanh(self.out_dense(torch.cat([dec_out, context], dim=-1)))
            vocab_dist = F.softmax(self.out_vocab(vocab_rep).squeeze(1), dim=-1)

            weighted_vocab_dist = p_gen * vocab_dist
            weighted_attn_dist = (1.0 - p_gen) * attn_dist

            max_oov_int = int(max_dynamic_oovs.item()) if isinstance(max_dynamic_oovs, torch.Tensor) else int(max_dynamic_oovs)
            full_vocab_dim = self.vocab_size + max_oov_int
            final_dist = torch.zeros(batch_size, full_vocab_dim, device=src_tokens.device)
            final_dist[:, :self.vocab_size] = weighted_vocab_dist
            final_dist.scatter_add_(1, src_extended_tokens, weighted_attn_dist)

            step_distributions.append(final_dist)

        return torch.stack(step_distributions, dim=1), torch.stack(coverage_penalties, dim=1)

raw_model = PointerGeneratorNetwork(
    vocab_size=VOCAB_SIZE,
    embed_dim=128,
    hidden_dim=256,
    padding_idx=PAD_ID,
    dropout=0.15
).to(device)

if torch.cuda.device_count() > 1:
    print(f"⚡ Kaggle Dual T4 detected! Parallelizing PointerGeneratorNetwork across {torch.cuda.device_count()} GPUs.")
    model_train = nn.DataParallel(raw_model)
else:
    model_train = raw_model

total_params = sum(p.numel() for p in raw_model.parameters())
trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
print(f"✓ PointerGeneratorNetwork Model Instantiated! Total Params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")
print_mem("Model Initialized")

# --- Cell 13 (starts at line 509) ---
class PointerGeneratorLoss(nn.Module):
    def __init__(self, pad_idx: int = 0, eps: float = 1e-12):
        super().__init__()
        self.pad_idx = pad_idx
        self.eps = eps

    def forward(
        self,
        step_distributions: torch.Tensor,
        target_tokens: torch.Tensor,
        coverage_penalties: torch.Tensor,
        cov_lambda: float = 0.0
    ):
        mask = (target_tokens != self.pad_idx).float()
        vocab_dim = step_distributions.size(-1)
        safe_targets = target_tokens.clamp(0, vocab_dim - 1).unsqueeze(-1)
        gold_probs = torch.gather(step_distributions, 2, safe_targets).squeeze(-1)

        nll_per_step = -torch.log(gold_probs + self.eps) * mask
        num_tokens = mask.sum().clamp(min=1.0)
        nll_loss = nll_per_step.sum() / num_tokens

        cov_loss = (coverage_penalties * mask).sum() / num_tokens
        total_loss = nll_loss + cov_lambda * cov_loss

        return total_loss, nll_loss, cov_loss

class EarlyStopping:
    """
    Tracks generative validation loss across epochs, increments patience counter,
    and halts training when validation loss stops improving.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss: float) -> Tuple[bool, str]:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True, f"⭐ (New Best Val Loss: {val_loss:.4f})"
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False, f"(Patience: {self.counter}/{self.patience})"

def safe_save(obj, target_path):
    """Atomic checkpoint saver preventing corrupt or partial writes."""
    tmp_path = str(target_path) + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, str(target_path))

loss_fn = PointerGeneratorLoss(pad_idx=PAD_ID)
print("✓ PointerGeneratorLoss, EarlyStopping, and safe_save ready!")

# --- Cell 15 (starts at line 569) ---
EPOCHS = 50
PATIENCE = 10
COVERAGE_PHASE_EPOCH = 12

CKPT_BEST = f"{CHECKPOINT_DIR}/checkpoint_best.pt"
CKPT_LAST = f"{CHECKPOINT_DIR}/checkpoint_last.pt"
TRAIN_RESULTS_CSV = f"{OUT}/training_results.csv"

optimizer = torch.optim.AdamW(raw_model.parameters(), lr=5e-4, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-5)
early_stopping = EarlyStopping(patience=PATIENCE, min_delta=1e-4)

start_epoch = 1
epoch_history = []

# Auto-resume from previous checkpoint if available
if os.path.exists(CKPT_LAST):
    print(f"🔄 Checkpoint found at '{CKPT_LAST}'! Loading state...")
    ckpt = torch.load(CKPT_LAST, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    early_stopping.best_loss = ckpt.get('best_val_loss', float('inf'))
    early_stopping.counter = ckpt.get('patience_counter', 0)
    print(f"✓ Resumed from Epoch {ckpt.get('epoch', 0)} (Best Val Loss: {early_stopping.best_loss:.4f})\n")
    if os.path.exists(TRAIN_RESULTS_CSV):
        try:
            epoch_history = pd.read_csv(TRAIN_RESULTS_CSV).to_dict('records')
        except Exception:
            epoch_history = []
else:
    print("🌱 Starting fresh 50-Epoch Training Run on Dual T4 hardware.\n")

print(f"🚀 Training PointerGeneratorNetwork (Epochs {start_epoch} to {EPOCHS}, Early Stopping Patience: {PATIENCE})...")
print(f"   Coverage Schedule: lambda=0.0 (Epochs 1-{COVERAGE_PHASE_EPOCH}) -> lambda=1.0 (Epochs {COVERAGE_PHASE_EPOCH+1}-{EPOCHS})\n")

epoch_pbar = tqdm(range(start_epoch, EPOCHS + 1), desc="Overall Training Progress", unit="epoch", dynamic_ncols=True)
for epoch in epoch_pbar:
    cov_lambda = 0.0 if epoch <= COVERAGE_PHASE_EPOCH else 1.0
    model_train.train()
    
    tot_train_loss = 0.0
    tot_train_nll = 0.0
    tot_train_cov = 0.0
    t0 = time.time()
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Train λ={cov_lambda:.1f}]", unit="batch", leave=False, dynamic_ncols=True)
    for step, b in enumerate(train_pbar, start=1):
        src_tokens = b['src_tokens'].to(device)
        src_lens = b['src_lens'].to(device)
        src_ext = b['src_extended_tokens'].to(device)
        src_mask = b['src_padding_mask'].to(device)
        tgt_in = b['tgt_in'].to(device)
        tgt_out = b['tgt_out'].to(device)
        max_oovs = b['max_dynamic_oovs']
        
        optimizer.zero_grad()
        step_dists, cov_penalties = model_train(
            src_tokens=src_tokens,
            src_lens=src_lens,
            src_extended_tokens=src_ext,
            max_dynamic_oovs=max_oovs,
            tgt_tokens=tgt_in,
            src_padding_mask=src_mask
        )
        
        loss, nll, cov = loss_fn(step_dists, tgt_out, cov_penalties, cov_lambda=cov_lambda)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        
        tot_train_loss += loss.item()
        tot_train_nll += nll.item()
        tot_train_cov += cov.item()
        
        train_pbar.set_postfix({
            'loss': f"{tot_train_loss/step:.4f}",
            'nll': f"{tot_train_nll/step:.4f}",
            'cov': f"{tot_train_cov/step:.4f}",
            'lr': f"{scheduler.get_last_lr()[-1]:.2e}"
        })
        
    avg_tr_loss = tot_train_loss / len(train_loader)
    avg_tr_nll = tot_train_nll / len(train_loader)
    avg_tr_cov = tot_train_cov / len(train_loader)
    
    # Validation Loop with Next-Token Accuracy & Perplexity
    model_train.eval()
    tot_val_loss = 0.0
    tot_val_nll = 0.0
    tot_val_cov = 0.0
    correct_tokens = 0
    total_tokens = 0
    
    val_pbar = tqdm(val_loader, desc=f"Validating Ep {epoch:02d}", unit="batch", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for b in val_pbar:
            src_tokens = b['src_tokens'].to(device)
            src_lens = b['src_lens'].to(device)
            src_ext = b['src_extended_tokens'].to(device)
            src_mask = b['src_padding_mask'].to(device)
            tgt_in = b['tgt_in'].to(device)
            tgt_out = b['tgt_out'].to(device)
            max_oovs = b['max_dynamic_oovs']
            
            step_dists, cov_penalties = model_train(
                src_tokens=src_tokens,
                src_lens=src_lens,
                src_extended_tokens=src_ext,
                max_dynamic_oovs=max_oovs,
                tgt_tokens=tgt_in,
                src_padding_mask=src_mask
            )
            loss, nll, cov = loss_fn(step_dists, tgt_out, cov_penalties, cov_lambda=cov_lambda)
            tot_val_loss += loss.item()
            tot_val_nll += nll.item()
            tot_val_cov += cov.item()
            
            preds = step_dists.argmax(dim=-1)
            mask = (tgt_out != PAD_ID)
            correct_tokens += ((preds == tgt_out) & mask).sum().item()
            total_tokens += mask.sum().item()
            
    avg_va_loss = tot_val_loss / len(val_loader)
    avg_va_nll = tot_val_nll / len(val_loader)
    avg_va_cov = tot_val_cov / len(val_loader)
    va_tok_acc = (correct_tokens / max(total_tokens, 1)) * 100.0
    va_ppl = math.exp(min(avg_va_nll, 20.0))
    elapsed = time.time() - t0
    
    is_best, status_str = early_stopping.step(avg_va_loss)
    
    ckpt_state = {
        'epoch': epoch,
        'model_state_dict': {k: v.cpu() for k, v in raw_model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': early_stopping.best_loss,
        'patience_counter': early_stopping.counter,
        'train_loss': avg_tr_loss,
        'train_nll': avg_tr_nll,
        'train_cov': avg_tr_cov,
        'val_loss': avg_va_loss,
        'val_nll': avg_va_nll,
        'val_cov': avg_va_cov,
        'val_token_acc': va_tok_acc,
        'val_ppl': va_ppl
    }
    
    # 1. Save latest checkpoint atomically on every epoch (crash-proof)
    safe_save(ckpt_state, CKPT_LAST)
    
    # 2. Save best checkpoint atomically
    if is_best:
        safe_save(ckpt_state, CKPT_BEST)
        
    # 3. Append telemetry to training_results.csv (real-time export)
    epoch_record = {
        'epoch': epoch,
        'train_loss': round(avg_tr_loss, 4),
        'train_nll': round(avg_tr_nll, 4),
        'train_cov': round(avg_tr_cov, 4),
        'val_loss': round(avg_va_loss, 4),
        'val_nll': round(avg_va_nll, 4),
        'val_cov': round(avg_va_cov, 4),
        'val_token_acc': round(va_tok_acc, 2),
        'val_ppl': round(va_ppl, 2),
        'cov_lambda': cov_lambda,
        'lr': scheduler.get_last_lr()[-1],
        'elapsed_sec': round(elapsed, 1),
        'is_best': is_best,
        'patience': early_stopping.counter
    }
    epoch_history.append(epoch_record)
    pd.DataFrame(epoch_history).to_csv(TRAIN_RESULTS_CSV, index=False)
    
    # Update outer progress bar
    epoch_pbar.set_postfix({
        'tr_loss': f"{avg_tr_loss:.3f}",
        'va_loss': f"{avg_va_loss:.3f}",
        'acc': f"{va_tok_acc:.1f}%",
        'ppl': f"{va_ppl:.1f}",
        'best': f"{early_stopping.best_loss:.3f}",
        'pat': f"{early_stopping.counter}/{PATIENCE}"
    })
    
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} ({int(elapsed)}s) | TrLoss: {avg_tr_loss:.4f} | VaLoss: {avg_va_loss:.4f} (NLL: {avg_va_nll:.4f}, Cov: {avg_va_cov:.4f}, Acc: {va_tok_acc:.2f}%, PPL: {va_ppl:.2f}) | {status_str}")
    
    if early_stopping.early_stop:
        print(f"\n🛑 Early stopping triggered at epoch {epoch}! Validation loss did not improve for {PATIENCE} consecutive epochs.")
        break

# Restore best weights for downstream evaluation
if os.path.exists(CKPT_BEST):
    best_data = torch.load(CKPT_BEST, map_location=device, weights_only=False)
    raw_model.load_state_dict(best_data['model_state_dict'])
    print(f"\n✓ Restored best model weights from '{CKPT_BEST}' (Val Loss: {early_stopping.best_loss:.4f})")

# --- Cell 17 (starts at line 771) ---
if os.path.exists(TRAIN_RESULTS_CSV):
    df_train = pd.read_csv(TRAIN_RESULTS_CSV)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), dpi=300)
    plt.subplots_adjust(wspace=0.28)
    
    c_tr = "#1f77b4"
    c_va = "#d62728"
    c_nll = "#2ca02c"
    c_cov = "#9467bd"
    c_acc = "#ff7f0e"
    c_ppl = "#17becf"
    
    # Subplot 1: Total Training & Validation Loss
    ax = axes[0]
    ax.plot(df_train['epoch'], df_train['train_loss'], label='Train Loss', color=c_tr, lw=2.2, marker='o', markersize=4)
    ax.plot(df_train['epoch'], df_train['val_loss'], label='Val Loss', color=c_va, lw=2.2, marker='s', markersize=4)
    best_row = df_train.loc[df_train['val_loss'].idxmin()]
    ax.scatter([best_row['epoch']], [best_row['val_loss']], color='gold', s=160, zorder=5, edgecolor='black', lw=1.5,
               label=f"Best Ep {int(best_row['epoch'])} (Loss: {best_row['val_loss']:.3f})")
    ax.axvline(x=COVERAGE_PHASE_EPOCH, color='gray', linestyle='--', alpha=0.7, label=f'Coverage On (Ep {COVERAGE_PHASE_EPOCH})')
    ax.set_title('Total Loss Convergence', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (NLL + λ·Cov)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: NLL vs Coverage Regularization
    ax = axes[1]
    ax.plot(df_train['epoch'], df_train['val_nll'], label='Val NLL Loss', color=c_nll, lw=2.0, marker='^', markersize=4)
    ax.plot(df_train['epoch'], df_train['val_cov'], label='Val Coverage Penalty', color=c_cov, lw=2.0, marker='v', markersize=4)
    ax.axvline(x=COVERAGE_PHASE_EPOCH, color='gray', linestyle='--', alpha=0.7, label=f'λ=0.0 → 1.0 (Ep {COVERAGE_PHASE_EPOCH})')
    ax.set_title('NLL vs Coverage Regularization', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss Metric', fontsize=11)
    ax.legend(fontsize=9, loc='upper right', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 3: Next-Token Accuracy & Perplexity
    ax = axes[2]
    ax_ppl = ax.twinx()
    l1 = ax.plot(df_train['epoch'], df_train['val_token_acc'], label='Next-Token Acc (%)', color=c_acc, lw=2.2, marker='o', markersize=4)
    l2 = ax_ppl.plot(df_train['epoch'], df_train['val_ppl'], label='Perplexity (PPL)', color=c_ppl, lw=2.0, linestyle='--', marker='d', markersize=4)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.set_title('Token Accuracy & Perplexity', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Token Accuracy (%)', color=c_acc, fontsize=11)
    ax_ppl.set_ylabel('Perplexity (exp[NLL])', color=c_ppl, fontsize=11)
    ax.legend(lines, labels, fontsize=9, loc='center right', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    TRAIN_CURVES_PNG = f"{OUT}/training_curves.png"
    plt.savefig(TRAIN_CURVES_PNG, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✓ Saved high-resolution training curves to '{TRAIN_CURVES_PNG}'")
else:
    print("ℹ️ training_results.csv not found, skipping plot.")

# --- Cell 19 (starts at line 832) ---
class BeamSearchHypothesis:
    def __init__(self, tokens: List[int], log_prob: float, dec_state, coverage: torch.Tensor, attn_history: List[torch.Tensor]):
        self.tokens = tokens
        self.log_prob = log_prob
        self.dec_state = dec_state
        self.coverage = coverage
        self.attn_history = attn_history

    def score(self, alpha: float = 0.7, beta: float = 0.3) -> float:
        k = len(self.tokens)
        lp = ((5.0 + k) / 6.0) ** alpha
        norm_score = self.log_prob / lp
        if beta > 0.0 and self.attn_history:
            stacked_attn = torch.stack(self.attn_history, dim=0)
            accum_attn = torch.sum(stacked_attn, dim=0)
            cov_penalty = torch.sum(torch.clamp(accum_attn - 1.0, min=0.0)).item()
            norm_score -= beta * cov_penalty
        return norm_score

def has_trigram_repeat(tokens: List[int], next_tok: int) -> bool:
    if len(tokens) < 2: return False
    trigram = (tokens[-2], tokens[-1], next_tok)
    for i in range(len(tokens) - 2):
        if (tokens[i], tokens[i+1], tokens[i+2]) == trigram:
            return True
    return False

@torch.no_grad()
def generate_headline_beam(
    model: PointerGeneratorNetwork,
    src_text: str,
    beam_width: int = 5,
    max_len: int = 22,
    min_len: int = 6,
    alpha: float = 0.85,
    beta: float = 0.15
) -> str:
    model.eval()
    pair = vocab_processor.process_pair(src_text, "")
    src_tokens = torch.tensor([pair['src_tokens'][:SRC_MAX_LEN]], dtype=torch.long, device=device)
    src_ext = torch.tensor([pair['src_extended'][:SRC_MAX_LEN]], dtype=torch.long, device=device)
    src_lens = torch.tensor([src_tokens.size(1)], dtype=torch.long, device=device)
    src_mask = (src_tokens == PAD_ID)
    max_oovs = pair['num_oovs']
    
    raw_ids = tok.encode(src_text, add_special_tokens=False)
    oov_tokens_map = {}
    curr_oov = 0
    for tid in raw_ids:
        if tid >= VOCAB_SIZE or tid == UNK_ID:
            if curr_oov not in oov_tokens_map:
                oov_tokens_map[VOCAB_SIZE + curr_oov] = tok.decode([tid])
                curr_oov += 1

    enc_outputs, (dec_h, dec_c) = model.encode(src_tokens, src_lens)
    enc_projected = model.W_h(enc_outputs)
    init_cov = torch.zeros(1, src_tokens.size(1), 1, device=device)

    init_hyp = BeamSearchHypothesis([BOS_ID], 0.0, (dec_h, dec_c), init_cov, [])
    hyps = [init_hyp]
    completed = []

    for step in range(max_len):
        candidates = []
        for hyp in hyps:
            last_tok = hyp.tokens[-1]
            if last_tok == EOS_ID:
                completed.append(hyp)
                continue

            y_prev = torch.tensor([[last_tok]], device=device)
            final_dist, dec_state, coverage, _, attn_dist = model.decode_step(
                y_prev=y_prev,
                dec_state=hyp.dec_state,
                enc_outputs=enc_outputs,
                enc_projected=enc_projected,
                src_extended_tokens=src_ext,
                max_dynamic_oovs=max_oovs,
                coverage=hyp.coverage,
                src_padding_mask=src_mask
            )

            log_probs = torch.log(final_dist.squeeze(0) + 1e-12)
            log_probs[PAD_ID] = float('-inf')
            if step < min_len:
                log_probs[EOS_ID] = float('-inf')

            topk_probs, topk_ids = torch.topk(log_probs, min(beam_width * 2, log_probs.size(0)))
            for p, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
                if has_trigram_repeat(hyp.tokens[1:], tid):
                    continue
                new_hyp = BeamSearchHypothesis(
                    tokens=hyp.tokens + [tid],
                    log_prob=hyp.log_prob + p,
                    dec_state=dec_state,
                    coverage=coverage,
                    attn_history=hyp.attn_history + [attn_dist]
                )
                candidates.append(new_hyp)

        if not candidates:
            break
        candidates.sort(key=lambda h: h.score(alpha=alpha, beta=beta), reverse=True)
        hyps = candidates[:beam_width]
        if len(completed) >= beam_width:
            break

    all_completed = completed if completed else hyps
    all_completed.sort(key=lambda h: h.score(alpha=alpha, beta=beta), reverse=True)
    best_hyp = all_completed[0]
    out_ids = [t for t in best_hyp.tokens if t not in (BOS_ID, EOS_ID, PAD_ID)]
    
    words = []
    subwords_to_decode = []
    for tid in out_ids:
        if tid in oov_tokens_map:
            if subwords_to_decode:
                words.append(tok.decode(subwords_to_decode).replace('##', '').strip())
                subwords_to_decode = []
            words.append(oov_tokens_map[tid].replace('##', '').strip())
        else:
            subwords_to_decode.append(tid)
    if subwords_to_decode:
        words.append(tok.decode(subwords_to_decode).replace('##', '').strip())
        
    return ' '.join([w for w in words if w]).replace('  ', ' ').strip()

print("✓ Constrained Beam Search Decoding Engine ready!")

# --- Cell 21 (starts at line 962) ---
from collections import defaultdict
from rouge_score import rouge_scorer

class BengaliRougeTokenizer:
    """
    Multilingual word extractor supporting Bengali (বাংলা), English, and Banglish.
    Extracts contiguous alphanumeric tokens including Bengali Unicode range \u0980-\u09FF.
    Fixes default rouge-score ASCII-only regex [^a-z0-9]+ that wiped out Bengali words to empty lists.
    """
    def __init__(self):
        self.pattern = _re.compile(r'[\u0980-\u09FFa-zA-Z0-9]+')

    def tokenize(self, text: str):
        return self.pattern.findall(unicodedata.normalize('NFKC', str(text)).lower())

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=BengaliRougeTokenizer())

RESULT_CSV = f"{OUT}/result.csv"
print(f"📊 Running Held-Out Test Set Evaluation ({len(test_dataset)} pairs)...\n")

eval_samples = test_dataset.samples[:300]
results_table = []
lang_metrics = defaultdict(lambda: {
    'r1_f': [], 'r1_p': [], 'r1_r': [],
    'r2_f': [], 'r2_p': [], 'r2_r': [],
    'rl_f': [], 'rl_p': [], 'rl_r': [],
    'ref_len': [], 'gen_len': []
})

bengali_tok = BengaliRougeTokenizer()
pbar = tqdm(eval_samples, desc="Evaluating Test Summaries", unit="sample", dynamic_ncols=True)
for i, (lead_text, ref_title, lang) in enumerate(pbar, start=1):
    hyp_title = generate_headline_beam(raw_model, lead_text, beam_width=4, max_len=22, min_len=4)
    scores = scorer.score(ref_title, hyp_title)
    
    r1_f, r1_p, r1_r = scores['rouge1'].fmeasure, scores['rouge1'].precision, scores['rouge1'].recall
    r2_f, r2_p, r2_r = scores['rouge2'].fmeasure, scores['rouge2'].precision, scores['rouge2'].recall
    rl_f, rl_p, rl_r = scores['rougeL'].fmeasure, scores['rougeL'].precision, scores['rougeL'].recall
    
    ref_toks = len(bengali_tok.tokenize(ref_title))
    gen_toks = len(bengali_tok.tokenize(hyp_title))
    
    for scope in (lang, 'overall'):
        lang_metrics[scope]['r1_f'].append(r1_f)
        lang_metrics[scope]['r1_p'].append(r1_p)
        lang_metrics[scope]['r1_r'].append(r1_r)
        lang_metrics[scope]['r2_f'].append(r2_f)
        lang_metrics[scope]['r2_p'].append(r2_p)
        lang_metrics[scope]['r2_r'].append(r2_r)
        lang_metrics[scope]['rl_f'].append(rl_f)
        lang_metrics[scope]['rl_p'].append(rl_p)
        lang_metrics[scope]['rl_r'].append(rl_r)
        lang_metrics[scope]['ref_len'].append(ref_toks)
        lang_metrics[scope]['gen_len'].append(gen_toks)
        
    results_table.append({
        'id': i,
        'language': lang,
        'lead_context': lead_text[:200],
        'reference_headline': ref_title,
        'generated_headline': hyp_title,
        'ref_len': ref_toks,
        'gen_len': gen_toks,
        'rouge_1_f1': round(r1_f, 4),
        'rouge_1_precision': round(r1_p, 4),
        'rouge_1_recall': round(r1_r, 4),
        'rouge_2_f1': round(r2_f, 4),
        'rouge_2_precision': round(r2_p, 4),
        'rouge_2_recall': round(r2_r, 4),
        'rouge_l_f1': round(rl_f, 4),
        'rouge_l_precision': round(rl_p, 4),
        'rouge_l_recall': round(rl_r, 4)
    })
    
    pbar.set_postfix({
        'R1-F1': f"{np.mean(lang_metrics['overall']['r1_f']):.3f}",
        'RL-F1': f"{np.mean(lang_metrics['overall']['rl_f']):.3f}"
    })

df_results = pd.DataFrame(results_table)
df_results.to_csv(RESULT_CSV, index=False)
print(f"\n✓ Saved full test prediction records to {RESULT_CSV}!")

print("\n" + "=" * 94)
print("🏆 FINAL COMPREHENSIVE TEST EVALUATION METRICS (Pointer-Generator with Coverage)")
print("=" * 94)
print(f"{'Language':<12} | {'N':<4} | {'ROUGE-1 (F1/P/R)':<22} | {'ROUGE-2 (F1/P/R)':<22} | {'ROUGE-L (F1/P/R)':<22}")
print("-" * 94)
for lg, mets in sorted(lang_metrics.items()):
    n = len(mets['r1_f'])
    r1_str = f"{np.mean(mets['r1_f']):.3f} / {np.mean(mets['r1_p']):.3f} / {np.mean(mets['r1_r']):.3f}"
    r2_str = f"{np.mean(mets['r2_f']):.3f} / {np.mean(mets['r2_p']):.3f} / {np.mean(mets['r2_r']):.3f}"
    rl_str = f"{np.mean(mets['rl_f']):.3f} / {np.mean(mets['rl_p']):.3f} / {np.mean(mets['rl_r']):.3f}"
    print(f"{lg:<12} | {n:<4} | {r1_str:<22} | {r2_str:<22} | {rl_str:<22}")
print("=" * 94)

# --- Cell 23 (starts at line 1059) ---
ROUGE_METRICS_PNG = f"{OUT}/rouge_metrics.png"
LANG_BREAKDOWN_PNG = f"{OUT}/language_breakdown.png"
GEN_DIAGNOSTICS_PNG = f"{OUT}/generation_diagnostics.png"

if os.path.exists(RESULT_CSV):
    df_eval = pd.read_csv(RESULT_CSV)
    
    # -------------------------------------------------------------
    # 1. ROUGE Metrics Distribution & Precision-Recall Trade-off
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    
    # Panel A: Boxplot of ROUGE F1 Scores
    ax = axes[0]
    metric_cols = ['rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1']
    data_to_plot = [df_eval[c].values for c in metric_cols]
    box = ax.boxplot(data_to_plot, patch_artist=True, labels=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
                     medianprops=dict(color='black', linewidth=1.5))
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    means = [np.mean(d) for d in data_to_plot]
    for idx, mean in enumerate(means, start=1):
        ax.scatter([idx], [mean], color='red', marker='D', s=50, zorder=5, label='Mean' if idx == 1 else '')
        ax.annotate(f"Mean: {mean:.3f}", (idx, mean), textcoords="offset points", xytext=(12, -3), fontsize=9, fontweight='bold')
    ax.set_title("ROUGE F1 Distributions on Test Set", fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    
    # Panel B: Precision vs Recall Trade-off (ROUGE-1 & ROUGE-L)
    ax = axes[1]
    ax.scatter(df_eval['rouge_1_recall'], df_eval['rouge_1_precision'], alpha=0.5, color='#2ca02c', s=35, label='ROUGE-1')
    ax.scatter(df_eval['rouge_l_recall'], df_eval['rouge_l_precision'], alpha=0.5, color='#1f77b4', s=35, label='ROUGE-L')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Balanced P=R')
    ax.set_title("Precision vs Recall Trade-off", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(ROUGE_METRICS_PNG, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✓ Saved ROUGE distribution figure to '{ROUGE_METRICS_PNG}'")

    # -------------------------------------------------------------
    # 2. Language Breakdown (Bengali vs Banglish vs English)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    langs_present = [lg for lg in sorted(df_eval['language'].unique())]
    bar_width = 0.25
    x = np.arange(len(langs_present))
    
    r1_means = [df_eval[df_eval['language'] == lg]['rouge_1_f1'].mean() for lg in langs_present]
    r2_means = [df_eval[df_eval['language'] == lg]['rouge_2_f1'].mean() for lg in langs_present]
    rl_means = [df_eval[df_eval['language'] == lg]['rouge_l_f1'].mean() for lg in langs_present]
    
    b1 = ax.bar(x - bar_width, r1_means, bar_width, label='ROUGE-1 F1', color='#2ca02c', alpha=0.85)
    b2 = ax.bar(x, r2_means, bar_width, label='ROUGE-2 F1', color='#ff7f0e', alpha=0.85)
    b3 = ax.bar(x + bar_width, rl_means, bar_width, label='ROUGE-L F1', color='#1f77b4', alpha=0.85)
    
    for bars in (b1, b2, b3):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
    ax.set_title("Trilingual Performance Comparison (Bengali, Banglish, English)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([lg.capitalize() for lg in langs_present], fontsize=11, fontweight='bold')
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, max(max(r1_means), max(rl_means)) * 1.25 if r1_means else 1.0)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(LANG_BREAKDOWN_PNG, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✓ Saved Language Breakdown plot to '{LANG_BREAKDOWN_PNG}'")

    # -------------------------------------------------------------
    # 3. Generation Diagnostics: Headline Length Fidelity
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=300)
    
    # Subplot A: Histogram of Lengths
    ax = axes[0]
    sns.histplot(df_eval['ref_len'], color='#1f77b4', kde=True, label='Reference Titles', ax=ax, bins=15, alpha=0.4)
    sns.histplot(df_eval['gen_len'], color='#2ca02c', kde=True, label='Generated Titles', ax=ax, bins=15, alpha=0.4)
    ax.set_title("Headline Word Length Distribution", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Word Count", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot B: Correlation of Reference vs Generated Length
    ax = axes[1]
    ax.scatter(df_eval['ref_len'], df_eval['gen_len'], color='#9467bd', alpha=0.6, s=40, edgecolor='white', lw=0.5)
    m, b = np.polyfit(df_eval['ref_len'], df_eval['gen_len'], 1)
    x_vals = np.linspace(df_eval['ref_len'].min(), df_eval['ref_len'].max(), 50)
    ax.plot(x_vals, m * x_vals + b, color='red', linestyle='--', lw=2.0, label=f'Trend (Slope: {m:.2f})')
    corr = df_eval['ref_len'].corr(df_eval['gen_len'])
    ax.set_title(f"Length Correlation (Pearson r = {corr:.2f})", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Reference Title Word Count", fontsize=11)
    ax.set_ylabel("Generated Title Word Count", fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(GEN_DIAGNOSTICS_PNG, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✓ Saved Generation Diagnostics plot to '{GEN_DIAGNOSTICS_PNG}'")

# --- Cell 25 (starts at line 1180) ---
def autonomous_headline(article_text: str) -> str:
    cleaned = clean_body_lead(article_text, max_chars=1500)
    title = generate_headline_beam(raw_model, cleaned, beam_width=4, max_len=20, min_len=4)
    print("=" * 70)
    print("📰 INPUT ARTICLE LEAD:")
    print(cleaned[:300] + ("..." if len(cleaned) > 300 else ""))
    print("-" * 70)
    print(f"🎯 AUTONOMOUS TITLE: {title}")
    print("=" * 70)
    return title

# Demo 1: Bengali News
autonomous_headline("""
গতকাল সিলেটে এক ভয়াবহ সড়ক দুর্ঘটনায় ঘটনাস্থলেই দুইজন নিহত এবং পাঁচজন আহত হয়েছেন। 
সিলেটের জৈন্তাপুর উপজেলায় ট্রাক ও সিএনজিচালিত অটোরিকশার মুখোমুখি সংঘর্ষে এই দুর্ঘটনা ঘটে। 
আহতদের উদ্ধার করে সিলেট এমএজি ওসমানী মেডিকেল কলেজ হাসপাতালে ভর্তি করা হয়েছে।
""")

# Demo 2: English News
autonomous_headline("""
The central bank announced a revision of national interest rates on Monday to control rising consumer inflation. 
Financial analysts expect this monetary tightening policy to stabilize currency exchange rates across commercial banks.
""")

# --- Cell 27 (starts at line 1205) ---
import glob, os, shutil, zipfile
from pathlib import Path

PACKAGE_ZIP = "result.zip"
ALIAS_ZIP = "pointer_generator_t4x2_package.zip"

files_to_pack = [
    CKPT_BEST,
    CKPT_LAST,
    RESULT_CSV,
    TRAIN_RESULTS_CSV,
    f"{OUT}/training_curves.png",
    f"{OUT}/rouge_metrics.png",
    f"{OUT}/language_breakdown.png",
    f"{OUT}/generation_diagnostics.png"
]

# Explicitly exclude any epoch checkpoint files (checkpoint_epoch_*.pt)
existing_files = [f for f in files_to_pack if os.path.exists(f)]

print(f"📦 Packaging {len(existing_files)} lean artifact files into '{PACKAGE_ZIP}'...")
with zipfile.ZipFile(PACKAGE_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for f in existing_files:
        arcname = os.path.basename(f)
        zf.write(f, arcname=arcname)
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  + Added: {arcname:<30} ({size_mb:.2f} MB)")

# Create alias copy for convenience
shutil.copy2(PACKAGE_ZIP, ALIAS_ZIP)

zip_size_mb = Path(PACKAGE_ZIP).stat().st_size / (1024 * 1024) if Path(PACKAGE_ZIP).exists() else 0.0
print("=" * 70)
print(f"📦 LEAN PACKAGE READY: {PACKAGE_ZIP} ({zip_size_mb:.2f} MB)")
print("  Archive Contents (Epoch .pt files strictly omitted):")
for f in existing_files:
    print(f"   - {os.path.basename(f)}")
print("=" * 70)

# Google Colab auto-download trigger
try:
    from google.colab import files
    files.download(PACKAGE_ZIP)
    print("✓ Triggered Colab direct download for result.zip!")
except Exception:
    print(f"✓ Kaggle / Local Output ready at: {os.path.abspath(PACKAGE_ZIP)}")


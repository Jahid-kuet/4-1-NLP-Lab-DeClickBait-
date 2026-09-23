# --- Cell 2 (starts at line 1) ---
!nvidia-smi --query-gpu=name,memory.total --format=csv
# Install required libraries
!pip install -q rouge-score "transformers>=4.40.0" matplotlib seaborn pandas numpy
print("✓ Dependencies installed successfully!")

# --- Cell 3 (starts at line 7) ---
# =====================================================================
# GOOGLE DRIVE MOUNTING (SURVIVES COLAB DISCONNECTIONS & TIMEOUTS)
# =====================================================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('✓ Google Drive mounted successfully! Checkpoints will persist under MyDrive/declickbait/')
    DRIVE_AVAILABLE = True
except Exception as e:
    print(f'ℹ️ Running in standalone mode without Google Drive: {e}')
    DRIVE_AVAILABLE = False

# --- Cell 4 (starts at line 20) ---
import glob, json, math, os, random, re as _re, shutil, time, unicodedata, zipfile
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# Reproducibility seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ---------------------------------------------------------------------
# Dynamic Dataset & Path Discovery
# ---------------------------------------------------------------------
def find_data_root() -> str:
    possible_roots = [
        Path('splits'),
        Path('data/processed/splits'),
        Path('/content/drive/MyDrive/declickbait/splits'),
        Path('/content/splits'),
        Path('../splits'),
        Path('../../splits'),
        Path('data/splits')
    ]
    for p in possible_roots:
        if p.exists() and (p / 'train_train_val_test.jsonl').exists():
            print(f"✓ Located dataset directory at: {p.resolve()}")
            return str(p.resolve())

    # Check current directory JSONL files
    local_candidates = list(Path('.').glob('*train*.jsonl'))
    if local_candidates:
        p = local_candidates[0].parent
        print(f"✓ Located dataset directory at: {p.resolve()}")
        return str(p.resolve())

    # Auto-extract from zip archive if present
    zip_candidates = [
        'declickbait-splits.zip',
        '/content/declickbait-splits.zip',
        '/content/drive/MyDrive/declickbait/declickbait-splits.zip',
        '../declickbait-splits.zip'
    ]
    for zc in zip_candidates:
        if os.path.exists(zc):
            print(f"📦 Extracting dataset from archive '{zc}'...")
            with zipfile.ZipFile(zc, 'r') as z:
                z.extractall('.')
            if Path('splits/train_train_val_test.jsonl').exists():
                print("✓ Successfully extracted splits directory!")
                return str(Path('splits').resolve())

    print("⚠️ WARNING: splits directory not found automatically. Defaulting to local 'splits'.")
    return 'splits'

DATA_ROOT = find_data_root()

# Output directories on Drive (if available) or local disk
if DRIVE_AVAILABLE and Path('/content/drive/MyDrive').exists():
    OUT = '/content/drive/MyDrive/declickbait/outputs/11_decoupled_multihead'
else:
    OUT = 'outputs/11_decoupled_multihead'

CHECKPOINT_DIR = f"{OUT}/checkpoints"
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
print(f"✓ Checkpoints & results will persist under: {OUT}")

def print_mem(tag=''):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        res = torch.cuda.memory_reserved() / 1e9
        print(f"[{tag}] VRAM Allocated: {alloc:.2f} GB | Reserved: {res:.2f} GB / 15.0 GB")

# --- Cell 5 (starts at line 108) ---
import json, os, unicodedata, re as _re, pandas as pd
from pathlib import Path
from typing import Any

def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _re.sub(r'[\r\n\t]+', ' ', text)
    return _re.sub(r'\s+', ' ', text).strip()

def load_jsonl_dataset(split_name: str) -> pd.DataFrame:
    file_path = Path(DATA_ROOT) / f"{split_name}_train_val_test.jsonl"
    if not file_path.exists():
        candidates = list(Path(DATA_ROOT).glob(f"*{split_name}*.jsonl"))
        if candidates:
            file_path = candidates[0]
        else:
            raise FileNotFoundError(f"Cannot find split '{split_name}' in {DATA_ROOT}")
            
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Title extraction: prioritize 'original_title' used in dataset splits
            title = (
                item.get("original_title") or
                item.get("title") or
                item.get("headline") or
                item.get("clickbait_title") or
                ""
            )
            
            # Target truth extraction: prioritize 'target_title'
            truth = (
                item.get("target_title") or
                item.get("truth_title") or
                item.get("declickbaited_title") or
                item.get("clarified_title") or
                item.get("truth_headline") or
                ""
            )
            if not truth or str(truth).strip() == "":
                truth = title
                
            # Content lead text extraction: prioritize 'body' used in dataset splits
            content = (
                item.get("body") or
                item.get("lead_content") or
                item.get("content") or
                item.get("content_first_paragraph") or
                item.get("article") or
                ""
            )
            
            # Label extraction: prioritize 'clickbait' (0 or 1)
            if "clickbait" in item:
                label = float(item["clickbait"])
            elif "label" in item:
                label = float(item["label"])
            elif "is_clickbait" in item:
                label = 1.0 if item["is_clickbait"] else 0.0
            else:
                label = 1.0
                
            # Language extraction: prioritize 'language_mix'
            lang = item.get("language_mix") or item.get("language") or item.get("lang") or "bn"
            if lang in ("bengali", "bangla"):
                lang = "bn"
            elif lang == "banglish":
                lang = "be"
            elif lang == "english":
                lang = "en"
            
            t_clean = clean_text(title)
            c_clean = clean_text(content)
            y_clean = clean_text(truth)
            
            if t_clean and c_clean:
                records.append({
                    "title": t_clean,
                    "content": c_clean,
                    "target_title": y_clean,
                    "label": label,
                    "language": lang
                })
                
    df = pd.DataFrame(records)
    print(f"✓ Split '{split_name:<5}': Loaded {len(df):,d} valid instances (Clickbait: {(df['label']==1.0).sum():,d}, Non-Clickbait: {(df['label']==0.0).sum():,d})")
    return df

print("Loading Declickbait Dataset splits...")
df_train = load_jsonl_dataset("train")
df_val = load_jsonl_dataset("val")
df_test = load_jsonl_dataset("test")

# --- Cell 6 (starts at line 208) ---
# Prevent transformers generation module from triggering broken sklearn/scipy imports on NumPy 2
try:
    import transformers.utils.import_utils as _tiu
    _tiu._sklearn_available = False
except Exception:
    pass

from transformers import AutoTokenizer

TOKENIZER_NAME = "csebuetnlp/banglishbert"
print(f"Loading BUET multilingual tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

VOCAB_SIZE = tokenizer.vocab_size  # 64,768
PAD_ID = tokenizer.pad_token_id   # 0
UNK_ID = tokenizer.unk_token_id   # 1
BOS_ID = tokenizer.cls_token_id or tokenizer.bos_token_id  # 2 ([CLS])
EOS_ID = tokenizer.sep_token_id or tokenizer.eos_token_id  # 3 ([SEP])

print(f"✓ Tokenizer Ready: Vocab Size = {VOCAB_SIZE:,} | PAD={PAD_ID} | UNK={UNK_ID} | BOS={BOS_ID} | EOS={EOS_ID}")

# --- Cell 7 (starts at line 230) ---
from torch.utils.data import Dataset, DataLoader

MAX_TITLE_LEN = 64
MAX_CONTENT_LEN = 512
MAX_TARGET_LEN = 32

class ExtendedVocabBatchProcessor:
    """
    Maps out-of-vocabulary and rare tokens to dynamic extended indices (VOCAB_SIZE + i),
    enabling exact entity pointer-copying from article content into generated headlines.
    """
    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def process_pair(self, src_text: str, tgt_text: str = ""):
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
                tgt_mapped.append(oov_to_ext_id[tid])
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

vocab_processor = ExtendedVocabBatchProcessor(tokenizer, VOCAB_SIZE)

class MultiHeadDeclickbaitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab_proc: ExtendedVocabBatchProcessor, max_t=64, max_c=512, max_tgt=32):
        self.df = df.reset_index(drop=True)
        self.proc = vocab_proc
        self.max_t = max_t
        self.max_c = max_c
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        title_str = str(row['title'])
        content_str = str(row['content'])
        target_str = str(row['target_title'])
        label = float(row['label'])
        lang = str(row['language'])
        
        # Candidate title tokenization
        t_ids = self.proc.tokenizer.encode(title_str, add_special_tokens=False)[:self.max_t]
        
        # Lead content tokenization with extended OOV mapping
        p = self.proc.process_pair(content_str, target_str)
        c_tokens = p['src_tokens'][:self.max_c]
        c_extended = p['src_extended'][:self.max_c]
        
        # Target title tokenization: [BOS] + target + [EOS]
        tgt_tokens = [BOS_ID] + p['tgt_tokens'][:self.max_tgt - 2] + [EOS_ID]
        
        return {
            'title_ids': t_ids,
            'content_ids': c_tokens,
            'content_extended': c_extended,
            'target_ids': tgt_tokens,
            'num_oovs': p['num_oovs'],
            'label': label,
            'language': lang,
            'title_raw': title_str,
            'content_raw': content_str,
            'target_raw': target_str
        }

def multihead_collate_fn(batch):
    batch_size = len(batch)
    
    t_lens = [len(item['title_ids']) for item in batch]
    c_lens = [len(item['content_ids']) for item in batch]
    tgt_lens = [len(item['target_ids']) for item in batch]
    
    max_t = max(t_lens)
    max_c = max(c_lens)
    max_tgt = max(tgt_lens)
    max_oovs = max(item['num_oovs'] for item in batch)
    
    t_tensor = torch.full((batch_size, max_t), PAD_ID, dtype=torch.long)
    t_mask = torch.zeros((batch_size, max_t), dtype=torch.long)
    
    c_tensor = torch.full((batch_size, max_c), PAD_ID, dtype=torch.long)
    c_mask = torch.zeros((batch_size, max_c), dtype=torch.long)
    c_ext_tensor = torch.full((batch_size, max_c), PAD_ID, dtype=torch.long)
    
    tgt_tensor = torch.full((batch_size, max_tgt), PAD_ID, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.float)
    
    languages = []
    titles_raw = []
    contents_raw = []
    targets_raw = []
    
    for i, item in enumerate(batch):
        t_l = len(item['title_ids'])
        c_l = len(item['content_ids'])
        tgt_l = len(item['target_ids'])
        
        t_tensor[i, :t_l] = torch.tensor(item['title_ids'], dtype=torch.long)
        t_mask[i, :t_l] = 1
        
        c_tensor[i, :c_l] = torch.tensor(item['content_ids'], dtype=torch.long)
        c_mask[i, :c_l] = 1
        c_ext_tensor[i, :c_l] = torch.tensor(item['content_extended'], dtype=torch.long)
        
        tgt_tensor[i, :tgt_l] = torch.tensor(item['target_ids'], dtype=torch.long)
        labels[i] = item['label']
        
        languages.append(item['language'])
        titles_raw.append(item['title_raw'])
        contents_raw.append(item['content_raw'])
        targets_raw.append(item['target_raw'])
        
    return {
        'title_ids': t_tensor,
        'title_mask': t_mask,
        'content_ids': c_tensor,
        'content_mask': c_mask,
        'content_extended': c_ext_tensor,
        'target_ids': tgt_tensor,
        'labels': labels,
        'max_oovs': max_oovs,
        'languages': languages,
        'titles_raw': titles_raw,
        'contents_raw': contents_raw,
        'targets_raw': targets_raw
    }

# DataLoaders: Batch size 16 is optimal for Colab Free T4 16GB VRAM
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

train_dataset = MultiHeadDeclickbaitDataset(df_train, vocab_processor, MAX_TITLE_LEN, MAX_CONTENT_LEN, MAX_TARGET_LEN)
val_dataset = MultiHeadDeclickbaitDataset(df_val, vocab_processor, MAX_TITLE_LEN, MAX_CONTENT_LEN, MAX_TARGET_LEN)
test_dataset = MultiHeadDeclickbaitDataset(df_test, vocab_processor, MAX_TITLE_LEN, MAX_CONTENT_LEN, MAX_TARGET_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=multihead_collate_fn, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=multihead_collate_fn, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=multihead_collate_fn, num_workers=0, pin_memory=True)

print(f"✓ DataLoaders initialized: Train Batches: {len(train_loader):,d} | Val Batches: {len(val_loader):,d} | Test Batches: {len(test_loader):,d}")

# --- Cell 8 (starts at line 403) ---
# =====================================================================
# UNIFIED 2x8 MULTI-HEAD DECLICKBAIT NEURAL NETWORK ARCHITECTURE
# (~28.85M Parameters: 02_v3 Cross-Attention Classifier + 08 Pointer-Generator)
# =====================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with learnable dropout."""
    def __init__(self, d_model: int = 128, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerTowerBlock(nn.Module):
    """Pre-LN Transformer Encoder Block."""
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(query=normed, key=normed, value=normed, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoderTower(nn.Module):
    """Dedicated L1 Transformer Encoder Tower with final LayerNorm."""
    def __init__(self, num_layers: int, d_model: int = 128, n_heads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerTowerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = ~mask.bool() if mask is not None else None
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.final_norm(x)


class TitleGuidedEvidenceCrossAttention(nn.Module):
    """Cross-Attention alignment module from 02_v3."""
    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        in_dim = d_model * 5 + 3
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        t_emb: torch.Tensor,
        t_mask: torch.Tensor,
        c_emb: torch.Tensor,
        c_mask: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        key_padding_mask = ~c_mask.bool() if c_mask is not None else None
        a_ev, attn_weights = self.cross_attn(
            query=t_emb, key=c_emb, value=c_emb,
            key_padding_mask=key_padding_mask, need_weights=return_diagnostics
        )

        t_weights = t_mask.unsqueeze(-1).float()
        c_weights = c_mask.unsqueeze(-1).float()

        h_title = (t_emb * t_weights).sum(1) / t_weights.sum(1).clamp_min(1.0)
        h_evidence = (a_ev * t_weights).sum(1) / t_weights.sum(1).clamp_min(1.0)
        h_content_global = (c_emb * c_weights).sum(1) / c_weights.sum(1).clamp_min(1.0)

        diff_evidence = (h_title - h_evidence).abs()
        prod_evidence = h_title * h_evidence
        cos_evidence = F.cosine_similarity(h_title, h_evidence, dim=-1).unsqueeze(-1)
        cos_global = F.cosine_similarity(h_title, h_content_global, dim=-1).unsqueeze(-1)

        tok_cos = F.cosine_similarity(t_emb, a_ev, dim=-1)
        tok_cos_masked = tok_cos.masked_fill(~t_mask.bool(), 1.0)
        min_tok_cos = tok_cos_masked.min(dim=-1).values.unsqueeze(-1)

        feat = torch.cat([h_title, h_evidence, h_content_global, diff_evidence, prod_evidence, cos_evidence, cos_global, min_tok_cos], dim=-1)
        fused = self.fuse(feat)

        diagnostics = {
            "cos_evidence": cos_evidence.squeeze(-1),
            "cos_global": cos_global.squeeze(-1),
            "min_token_cos": min_tok_cos.squeeze(-1),
            "tok_cos_per_token": tok_cos_masked,
            "attn_weights": attn_weights if return_diagnostics else None
        }
        return fused, diagnostics


# =====================================================================
# 2. POINTER-GENERATOR HEAD WITH COVERAGE (08 GENERATOR)
# =====================================================================

class BahdanauCoverageAttention(nn.Module):
    """Bahdanau additive attention with accumulated coverage."""
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # 512 -> 256
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=True)       # 256 -> 256
        self.W_c = nn.Linear(1, hidden_dim, bias=False)               # 1 -> 256
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
        enc_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = encoder_outputs.size()
        Wh_h = self.W_h(encoder_outputs)             # [B, S, H]
        Ws_s = self.W_s(decoder_state).unsqueeze(1)  # [B, 1, H]
        score_input = Wh_h + Ws_s
        if coverage is not None:
            Wc_c = self.W_c(coverage.unsqueeze(-1))   # [B, S, H]
            score_input = score_input + Wc_c

        scores = self.v(torch.tanh(score_input)).squeeze(-1)  # [B, S]
        if enc_mask is not None:
            scores = scores.masked_fill(~enc_mask.bool(), -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, 512]

        step_cov_loss = torch.zeros(B, device=decoder_state.device)
        if coverage is not None:
            step_cov_loss = torch.sum(torch.min(attn_weights, coverage), dim=-1)
            new_coverage = coverage + attn_weights
        else:
            new_coverage = attn_weights

        return context, attn_weights, new_coverage, step_cov_loss


class PointerGeneratorDecoderHead(nn.Module):
    """08 Pointer-Generator Decoder with soft p_gen switch and extended OOV scatter."""
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.cell = nn.LSTMCell(embed_dim + hidden_dim * 2, hidden_dim)  # [128 + 512, 256]
        self.attention = BahdanauCoverageAttention(hidden_dim=hidden_dim)

        self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim)

        self.p_gen_linear = nn.Linear(hidden_dim * 2 + hidden_dim + embed_dim, 1)  # [512 + 256 + 128, 1]
        self.out_dense = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)        # [512 + 256, 256]
        self.dropout = nn.Dropout(dropout)
        self.out_vocab = nn.Linear(hidden_dim, vocab_size)                         # [256, 64768]

    def init_decoder_state(self, enc_h_last: torch.Tensor, enc_c_last: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # enc_h_last: [2, B, H] -> cat to [B, 2*H]
        h_cat = torch.cat([enc_h_last[0], enc_h_last[1]], dim=-1)
        c_cat = torch.cat([enc_c_last[0], enc_c_last[1]], dim=-1)
        h_0 = torch.tanh(self.reduce_h(h_cat))
        c_0 = torch.tanh(self.reduce_c(c_cat))
        return h_0, c_0

    def forward_step(
        self,
        input_emb: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        encoder_outputs: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
        enc_mask: Optional[torch.Tensor] = None,
        src_extended_ids: Optional[torch.Tensor] = None,
        max_oovs: int = 0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        cell_input = torch.cat([input_emb, context], dim=-1)
        h_t, c_t = self.cell(cell_input, (h_prev, c_prev))

        ctx_t, attn_weights, new_coverage, step_cov_loss = self.attention(h_t, encoder_outputs, coverage, enc_mask)
        p_gen = torch.sigmoid(self.p_gen_linear(torch.cat([ctx_t, h_t, input_emb], dim=-1)))

        out_hidden = torch.tanh(self.out_dense(torch.cat([h_t, ctx_t], dim=-1)))
        out_hidden = self.dropout(out_hidden)
        vocab_dist = F.softmax(self.out_vocab(out_hidden), dim=-1) * p_gen

        if max_oovs > 0:
            B = input_emb.size(0)
            extra_zeros = torch.zeros((B, max_oovs), device=input_emb.device, dtype=input_emb.dtype)
            final_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)
        else:
            final_dist = vocab_dist

        if src_extended_ids is not None:
            attn_scaled = attn_weights * (1.0 - p_gen)
            final_dist = final_dist.scatter_add(dim=1, index=src_extended_ids, src=attn_scaled)

        return final_dist.clamp_min(1e-12), (h_t, c_t), ctx_t, attn_weights, new_coverage, step_cov_loss


# =====================================================================
# 3. UNIFIED 2x8 MULTI-HEAD ARCHITECTURE (~28.85M PARAMS)
# =====================================================================

class MultiHeadDeclickbait2x8(nn.Module):
    """
    Unified 2x8 Multi-Head Declickbait Architecture (~28.85M Parameters).
    Combines 02_v3 Title-Guided Cross-Attention Evidence Classifier with 08 BiLSTM Pointer-Generator Network.
    """
    def __init__(
        self,
        vocab_size: int = 64768,
        embed_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        d_model: int = 128,
        n_heads: int = 4,
        title_layers: int = 3,
        content_layers: int = 4,
        d_ff: int = 512,
        max_title_len: int = 64,
        max_content_len: int = 512,
        pad_idx: int = 0,
        dropout: float = 0.15
    ):
        super().__init__()
        if embed_dim is None:
            embed_dim = d_model
        if hidden_dim is None:
            hidden_dim = 256 if d_model >= 128 else d_model * 2

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.pad_idx = pad_idx

        # 1. Shared BUET Multilingual Embedding Table (~8.29M params)
        self.shared_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(max_title_len, max_content_len) + 16, dropout=dropout)

        # 2. Head 1 (02_v3 Classifier Towers & Cross-Attention, ~1.54M params)
        self.title_tower = TransformerEncoderTower(num_layers=title_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.content_tower = TransformerEncoderTower(num_layers=content_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.evidence_aligner = TitleGuidedEvidenceCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # 3. Head 2 (08 Generator BiLSTM Encoder + Decoder, ~19.02M params incl. output softmax)
        self.enc_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.decoder_head = PointerGeneratorDecoderHead(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_title_transformer(self, t_ids: torch.Tensor, t_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.title_tower(self.pos_encoder(self.shared_embedding(t_ids)), mask=t_mask)

    def encode_content_transformer(self, c_ids: torch.Tensor, c_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.content_tower(self.pos_encoder(self.shared_embedding(c_ids)), mask=c_mask)

    def encode_title(self, t_ids: torch.Tensor, t_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode_title_transformer(t_ids, t_mask)

    def encode_content(self, c_ids: torch.Tensor, c_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode_content_transformer(c_ids, c_mask)

    def encode_content_lstm(self, c_ids: torch.Tensor, c_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        c_emb = self.shared_embedding(c_ids)
        return self.enc_lstm(c_emb)

    def forward_classifier(
        self,
        t_ids: torch.Tensor,
        t_mask: Optional[torch.Tensor] = None,
        c_ids: Optional[torch.Tensor] = None,
        c_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Dict[str, Any]:
        """Dedicated forward pass for Head 1 (02_v3 Evidence Classifier)."""
        if t_mask is None:
            t_mask = (t_ids != self.pad_idx).long()
        if c_mask is None and c_ids is not None:
            c_mask = (c_ids != self.pad_idx).long()

        t_repr = self.encode_title_transformer(t_ids, t_mask)
        c_repr = self.encode_content_transformer(c_ids, c_mask)
        fused, diag = self.evidence_aligner(t_repr, t_mask, c_repr, c_mask, return_diagnostics)
        cls_logits = self.classifier_head(fused).squeeze(-1)
        cls_probs = torch.sigmoid(cls_logits)

        cls_loss = None
        if labels is not None:
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, labels.float()).unsqueeze(0)

        output = {
            "logits": cls_logits,
            "probs": cls_probs,
            "cls_loss": cls_loss,
            "total_loss": cls_loss,
            "cos_evidence": diag["cos_evidence"],
            "cos_global": diag["cos_global"],
            "min_token_cos": diag["min_token_cos"]
        }
        if return_diagnostics:
            output["tok_cos_per_token"] = diag["tok_cos_per_token"]
            output["attn_weights"] = diag["attn_weights"]
        return output

    def forward_generator(
        self,
        c_ids: torch.Tensor,
        c_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        src_extended_ids: Optional[torch.Tensor] = None,
        max_oovs: int = 0,
        coverage_lambda: float = 0.0
    ) -> Dict[str, Any]:
        """Dedicated forward pass for Head 2 (08 BiLSTM Pointer-Generator)."""
        if c_mask is None:
            c_mask = (c_ids != self.pad_idx).long()

        B, tgt_len = target_ids.size()
        c_emb_seq = self.shared_embedding(c_ids)
        enc_outputs, (enc_h, enc_c) = self.enc_lstm(c_emb_seq)

        h_t, c_t = self.decoder_head.init_decoder_state(enc_h, enc_c)
        context = torch.zeros(B, self.hidden_dim * 2, device=c_ids.device)
        coverage = torch.zeros(B, c_ids.size(1), device=c_ids.device)

        step_losses, cov_losses = [], []
        correct_toks, total_toks = 0, 0

        for step in range(tgt_len - 1):
            input_tok = target_ids[:, step].clamp(0, self.vocab_size - 1)
            input_emb = self.shared_embedding(input_tok)

            final_dist, (h_t, c_t), context, attn, coverage, step_cov = self.decoder_head.forward_step(
                input_emb, (h_t, c_t), context, enc_outputs, coverage, c_mask, src_extended_ids, max_oovs
            )

            target_step = target_ids[:, step + 1]
            target_probs = final_dist.gather(1, target_step.unsqueeze(1)).squeeze(1)
            nll_step = -torch.log(target_probs.clamp_min(1e-12))
            step_mask = (target_step != self.pad_idx).float()

            step_losses.append((nll_step * step_mask).sum() / step_mask.sum().clamp_min(1.0))
            cov_losses.append((step_cov * step_mask).sum() / step_mask.sum().clamp_min(1.0))

            pred_tokens = final_dist.argmax(dim=-1)
            correct_toks += ((pred_tokens == target_step) * step_mask.bool()).sum().item()
            total_toks += step_mask.sum().item()

        gen_loss = torch.stack(step_losses).mean().unsqueeze(0)
        cov_loss = torch.stack(cov_losses).mean().unsqueeze(0)
        token_acc = torch.tensor(
            [(correct_toks / max(total_toks, 1)) * 100.0],
            device=c_ids.device,
            dtype=torch.float32
        )
        total_loss = gen_loss + (coverage_lambda * cov_loss)

        return {
            "gen_loss": gen_loss,
            "cov_loss": cov_loss,
            "total_loss": total_loss,
            "token_acc": token_acc
        }

    def forward(
        self,
        t_ids: Optional[torch.Tensor] = None,
        t_mask: Optional[torch.Tensor] = None,
        c_ids: Optional[torch.Tensor] = None,
        c_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        src_extended_ids: Optional[torch.Tensor] = None,
        max_oovs: int = 0,
        coverage_lambda: float = 0.0,
        mode: Optional[str] = None,
        return_diagnostics: bool = False
    ) -> Dict[str, Any]:
        if mode == "classifier":
            return self.forward_classifier(t_ids, t_mask, c_ids, c_mask, labels, return_diagnostics)
        elif mode == "generator":
            return self.forward_generator(c_ids, c_mask, target_ids, src_extended_ids, max_oovs, coverage_lambda)
        elif t_ids is not None and target_ids is not None:
            cls_out = self.forward_classifier(t_ids, t_mask, c_ids, c_mask, labels, return_diagnostics)
            gen_out = self.forward_generator(c_ids, c_mask, target_ids, src_extended_ids, max_oovs, coverage_lambda)
            tot_loss = cls_out["cls_loss"] + gen_out["total_loss"] if cls_out["cls_loss"] is not None else gen_out["total_loss"]
            out = {**cls_out, **gen_out, "total_loss": tot_loss}
            return out
        elif t_ids is not None:
            return self.forward_classifier(t_ids, t_mask, c_ids, c_mask, labels, return_diagnostics)
        elif target_ids is not None:
            return self.forward_generator(c_ids, c_mask, target_ids, src_extended_ids, max_oovs, coverage_lambda)
        else:
            raise ValueError("Either t_ids or target_ids must be provided to MultiHeadDeclickbait2x8.forward()")


# Alias for backward compatibility
MultiHeadBPEPointerGenerator = MultiHeadDeclickbait2x8

# Alias for backward compatibility
MultiHeadBPEPointerGenerator = MultiHeadDeclickbait2x8

# Instantiate Production Model
raw_model = MultiHeadDeclickbait2x8(
    vocab_size=VOCAB_SIZE,
    embed_dim=128,
    hidden_dim=256,
    d_model=128,
    n_heads=4,
    title_layers=3,
    content_layers=4,
    d_ff=512,
    max_title_len=MAX_TITLE_LEN,
    max_content_len=MAX_CONTENT_LEN,
    pad_idx=PAD_ID,
    dropout=0.15
).to(device)

# Multi-GPU DataParallel Setup for Dual T4x2
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    num_devices = torch.cuda.device_count()
    print("=" * 70)
    print(f"🚀 MULTI-GPU DATAPARALLEL ENABLED ACROSS {num_devices} T4 GPUs!")
    for gi in range(num_devices):
        vram_gb = torch.cuda.get_device_properties(gi).total_memory / 1e9
        print(f"   [GPU {gi}]: {torch.cuda.get_device_name(gi)} ({vram_gb:.2f} GB VRAM)")
    print("=" * 70)
    model = nn.DataParallel(raw_model)
else:
    model = raw_model
    if torch.cuda.is_available():
        print(f"✓ Single GPU Active: {torch.cuda.get_device_name(0)}")

total_params = sum(p.numel() for p in raw_model.parameters())
trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)

print("=" * 70)
print("🏛️ MULTI-HEAD DECLICKBAIT 2x8 SYSTEM INSTANTIATED")
print(f"   Total Parameters:     {total_params / 1e6:.2f}M ({total_params:,})")
print(f"   Trainable Parameters: {trainable_params / 1e6:.2f}M ({trainable_params:,})")
print("   Footprint Profile:    Unified 2x8 dual-head architecture (~115 MB memory footprint)")
print("=" * 70)
print_mem("After Model Initialization")

# --- Cell 9 (starts at line 907) ---
def safe_save(obj, target_path: str):
    # Atomic checkpoint saver preventing partial or corrupted writes on Google Drive FUSE.
    tmp_path = str(target_path) + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, str(target_path))

class MultiTaskEarlyStopping:
    # Dual-Criterion Early Stopping & Checkpoint Selector.
    # Triggers 'NEW BEST MODEL' if:
    #   1. Classifier Macro-F1 improves (captures classification breakthroughs, e.g. 79.5% -> 82.3%)
    #   OR
    #   2. Combined Multi-Task Loss decreases significantly.
    # This guarantees that generator validation noise will NEVER discard a peak classifier.
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, min_f1_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.min_f1_delta = min_f1_delta
        self.best_score = float('inf')  # Best validation total loss
        self.best_f1 = 0.0              # Best validation Macro-F1
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss: float, val_f1: float) -> bool:
        f1_improved = (val_f1 > self.best_f1 + self.min_f1_delta)
        loss_improved = (val_loss < self.best_score - self.min_delta)

        if f1_improved or loss_improved:
            if loss_improved:
                self.best_score = val_loss
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

print("✓ safe_save and Dual-Criterion MultiTaskEarlyStopping initialized!")

# --- Cell 10 (starts at line 949) ---
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

EPOCHS = 50
PATIENCE = 10
COVERAGE_PHASE_EPOCH = 12

CKPT_BEST = f"{CHECKPOINT_DIR}/checkpoint_best.pt"
CKPT_LAST = f"{CHECKPOINT_DIR}/checkpoint_last.pt"
TRAIN_RESULTS_CSV = f"{OUT}/training_results.csv"

# Access base module if wrapped in DataParallel
raw_model = model.module if isinstance(model, nn.DataParallel) else model

# =========================================================================
# OPTION 1: DECOUPLED PARAMETER PARTITIONS & INDEPENDENT OPTIMIZERS
# =========================================================================
# Head 1 (02_v3 Classifier + Shared Multilingual Embeddings): ~9.83M parameters
cls_params = (
    list(raw_model.shared_embedding.parameters()) +
    list(raw_model.pos_encoder.parameters()) +
    list(raw_model.title_tower.parameters()) +
    list(raw_model.content_tower.parameters()) +
    list(raw_model.evidence_aligner.parameters()) +
    list(raw_model.classifier_head.parameters())
)

# Head 2 (08 BiLSTM Pointer-Generator Head): ~19.02M parameters
gen_params = (
    list(raw_model.enc_lstm.parameters()) +
    list(raw_model.decoder_head.parameters())
)

n_cls = sum(p.numel() for p in cls_params)
n_gen = sum(p.numel() for p in gen_params)
total_p = sum(p.numel() for p in raw_model.parameters())

print("=" * 70)
print("⚡ OPTION 1: DECOUPLED OPTIMIZATION CONFIGURATION")
print(f"   Head 1 Cls Parameters: {n_cls:,} ({n_cls/1e6:.2f}M) -> AdamW(lr=2e-4, wd=1e-4)")
print(f"   Head 2 Gen Parameters: {n_gen:,} ({n_gen/1e6:.2f}M) -> AdamW(lr=5e-4, wd=1e-6)")
print(f"   Total Architecture:     {total_p:,} ({total_p/1e6:.2f}M)")
if isinstance(model, nn.DataParallel):
    print(f"   Dual-GPU Mode:          Active across {torch.cuda.device_count()} GPUs")
print("=" * 70)

# Two Independent Optimizers & Schedulers
optimizer_cls = torch.optim.AdamW(cls_params, lr=2e-4, weight_decay=1e-4)
scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_cls, T_max=EPOCHS * len(train_loader), eta_min=1e-5
)

optimizer_gen = torch.optim.AdamW(gen_params, lr=5e-4, weight_decay=1e-6)
scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_gen, T_max=EPOCHS * len(train_loader), eta_min=1e-5
)

early_stopping = MultiTaskEarlyStopping(patience=PATIENCE, min_delta=1e-4, min_f1_delta=0.001)

start_epoch = 1
epoch_history = []

# Auto-resume from Google Drive / disk if checkpoint exists
if os.path.exists(CKPT_LAST):
    print(f"🔄 Checkpoint found at '{CKPT_LAST}'! Resuming decoupled training state...")
    ckpt = torch.load(CKPT_LAST, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_cls_state_dict' in ckpt and 'optimizer_gen_state_dict' in ckpt:
        optimizer_cls.load_state_dict(ckpt['optimizer_cls_state_dict'])
        optimizer_gen.load_state_dict(ckpt['optimizer_gen_state_dict'])
        if 'scheduler_cls_state_dict' in ckpt:
            scheduler_cls.load_state_dict(ckpt['scheduler_cls_state_dict'])
        if 'scheduler_gen_state_dict' in ckpt:
            scheduler_gen.load_state_dict(ckpt['scheduler_gen_state_dict'])
        print("✓ Successfully restored dual decoupled optimizer and scheduler states.")
    elif 'optimizer_state_dict' in ckpt:
        print("ℹ️ Checkpoint from single joint optimizer detected; model weights loaded, initialized fresh decoupled optimizers.")
    start_epoch = ckpt.get('epoch', 0) + 1
    early_stopping.best_score = ckpt.get('best_score', float('inf'))
    early_stopping.best_f1 = ckpt.get('best_f1', 0.0)
    early_stopping.counter = ckpt.get('patience_counter', 0)
    print(f"✓ Resumed from Epoch {ckpt.get('epoch', 0)} (Best Val Score: {early_stopping.best_score:.4f}, Best F1: {early_stopping.best_f1*100:.2f}%)")
    if os.path.exists(TRAIN_RESULTS_CSV):
        try:
            epoch_history = pd.read_csv(TRAIN_RESULTS_CSV).to_dict('records')
        except Exception:
            epoch_history = []
else:
    print("🌱 Starting fresh 50-Epoch Decoupled Multi-Head Training Run (Option 1: Two Independent Optimizers).")

print(f"🚀 Training MultiHeadDeclickbait2x8 (Epochs {start_epoch} to {EPOCHS}, Early Stopping Patience: {PATIENCE})...")

epoch_pbar = tqdm(range(start_epoch, EPOCHS + 1), desc="Overall Training Progress", dynamic_ncols=True, unit="epoch")

for epoch in epoch_pbar:
    t0 = time.time()
    cov_lambda = 1.0 if epoch >= COVERAGE_PHASE_EPOCH else 0.0
    
    # -------------------------------------------------------------
    # 1. TRAINING STEP (ALTERNATING DECOUPLED GRADIENT DESCENT)
    # -------------------------------------------------------------
    model.train()
    tr_total_loss, tr_cls_loss, tr_gen_loss, tr_cov_loss = 0.0, 0.0, 0.0, 0.0
    
    batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False, dynamic_ncols=True)
    for batch in batch_pbar:
        t_ids = batch['title_ids'].to(device)
        t_mask = batch['title_mask'].to(device)
        c_ids = batch['content_ids'].to(device)
        c_mask = batch['content_mask'].to(device)
        c_ext = batch['content_extended'].to(device)
        tgt_ids = batch['target_ids'].to(device)
        labels = batch['labels'].to(device)
        max_oovs = batch['max_oovs']
        
        # ---------------------------------------------------------
        # Step A: Optimize Classifier Head (Dual GPU Parallel Scatter)
        # ---------------------------------------------------------
        optimizer_cls.zero_grad()
        cls_out = model(
            t_ids=t_ids,
            t_mask=t_mask,
            c_ids=c_ids,
            c_mask=c_mask,
            labels=labels,
            mode="classifier"
        )
        cls_loss = cls_out['cls_loss'].mean()
        cls_loss.backward()
        nn.utils.clip_grad_norm_(cls_params, max_norm=1.0)
        optimizer_cls.step()
        scheduler_cls.step()
        
        # ---------------------------------------------------------
        # Step B: Optimize Generator Head (Dual GPU Parallel Scatter)
        # ---------------------------------------------------------
        optimizer_gen.zero_grad()
        gen_out = model(
            c_ids=c_ids,
            c_mask=c_mask,
            target_ids=tgt_ids,
            src_extended_ids=c_ext,
            max_oovs=max_oovs,
            coverage_lambda=cov_lambda,
            mode="generator"
        )
        gen_loss = gen_out['total_loss'].mean()
        gen_loss.backward()
        nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
        optimizer_gen.step()
        scheduler_gen.step()
        
        batch_loss = cls_loss.item() + gen_loss.item()
        tr_total_loss += batch_loss
        tr_cls_loss += cls_loss.item()
        tr_gen_loss += gen_out['gen_loss'].mean().item()
        tr_cov_loss += gen_out['cov_loss'].mean().item()
        
        batch_pbar.set_postfix({
            'loss': f"{batch_loss:.3f}",
            'cls': f"{cls_loss.item():.3f}",
            'gen': f"{gen_out['gen_loss'].mean().item():.3f}",
            'lr_c': f"{optimizer_cls.param_groups[0]['lr']:.2e}",
            'lr_g': f"{optimizer_gen.param_groups[0]['lr']:.2e}"
        })
        
    num_tr = len(train_loader)
    avg_tr_loss = tr_total_loss / num_tr
    avg_tr_cls = tr_cls_loss / num_tr
    avg_tr_gen = tr_gen_loss / num_tr
    avg_tr_cov = tr_cov_loss / num_tr
    
    # -------------------------------------------------------------
    # 2. VALIDATION STEP
    # -------------------------------------------------------------
    model.eval()
    va_total_loss, va_cls_loss, va_gen_loss, va_cov_loss = 0.0, 0.0, 0.0, 0.0
    all_preds_probs, all_labels = [], []
    
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch in val_pbar:
            t_ids = batch['title_ids'].to(device)
            t_mask = batch['title_mask'].to(device)
            c_ids = batch['content_ids'].to(device)
            c_mask = batch['content_mask'].to(device)
            c_ext = batch['content_extended'].to(device)
            tgt_ids = batch['target_ids'].to(device)
            labels = batch['labels'].to(device)
            max_oovs = batch['max_oovs']
            
            cls_out = model(
                t_ids=t_ids,
                t_mask=t_mask,
                c_ids=c_ids,
                c_mask=c_mask,
                labels=labels,
                mode="classifier"
            )
            gen_out = model(
                c_ids=c_ids,
                c_mask=c_mask,
                target_ids=tgt_ids,
                src_extended_ids=c_ext,
                max_oovs=max_oovs,
                coverage_lambda=cov_lambda,
                mode="generator"
            )
            
            cls_vloss = cls_out['cls_loss'].mean().item()
            gen_vloss = gen_out['gen_loss'].mean().item()
            cov_vloss = gen_out['cov_loss'].mean().item()
            tot_vloss = gen_out['total_loss'].mean().item()
            
            va_total_loss += (cls_vloss + tot_vloss)
            va_cls_loss += cls_vloss
            va_gen_loss += gen_vloss
            va_cov_loss += cov_vloss
            
            all_preds_probs.extend(cls_out['probs'].cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
    num_va = len(val_loader)
    avg_va_loss = va_total_loss / num_va
    avg_va_cls = va_cls_loss / num_va
    avg_va_gen = va_gen_loss / num_va
    avg_va_cov = va_cov_loss / num_va
    val_ppl = math.exp(min(avg_va_gen, 100))
    
    # Dynamic threshold search for Macro-F1
    all_preds_probs = np.array(all_preds_probs)
    all_labels = np.array(all_labels)
    
    best_tau = 0.5
    best_f1 = 0.0
    for tau_cand in np.linspace(0.2, 0.8, 31):
        cand_preds = (all_preds_probs >= tau_cand).astype(int)
        f1_cand = f1_score(all_labels, cand_preds, average='macro', zero_division=0)
        if f1_cand > best_f1:
            best_f1 = f1_cand
            best_tau = tau_cand
            
    val_acc = accuracy_score(all_labels, (all_preds_probs >= best_tau).astype(int)) * 100.0
    val_auc = roc_auc_score(all_labels, all_preds_probs) if len(np.unique(all_labels)) > 1 else 0.5
    
    elapsed = time.time() - t0
    
    # Early stopping check (Considers both F1 improvements and Loss drops!)
    improved = early_stopping.step(avg_va_loss, best_f1)
    
    ckpt_state = {
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_cls_state_dict': optimizer_cls.state_dict(),
        'optimizer_gen_state_dict': optimizer_gen.state_dict(),
        'scheduler_cls_state_dict': scheduler_cls.state_dict(),
        'scheduler_gen_state_dict': scheduler_gen.state_dict(),
        'best_score': early_stopping.best_score,
        'best_f1': early_stopping.best_f1,
        'best_tau': best_tau,
        'patience_counter': early_stopping.counter
    }
    
    # Safe atomic saves (Zero intermediate epoch pt files)
    safe_save(ckpt_state, CKPT_LAST)
    
    if improved:
        safe_save(ckpt_state, CKPT_BEST)
        trophy = "⭐ [NEW BEST MODEL]"
    else:
        trophy = ""
        
    # Append telemetry to training_results.csv
    epoch_record = {
        'epoch': epoch,
        'train_total_loss': round(avg_tr_loss, 4),
        'train_cls_loss': round(avg_tr_cls, 4),
        'train_gen_loss': round(avg_tr_gen, 4),
        'train_cov_loss': round(avg_tr_cov, 4),
        'val_total_loss': round(avg_va_loss, 4),
        'val_cls_loss': round(avg_va_cls, 4),
        'val_gen_loss': round(avg_va_gen, 4),
        'val_cov_loss': round(avg_va_cov, 4),
        'val_ppl': round(val_ppl, 2),
        'val_acc': round(val_acc, 2),
        'val_macro_f1': round(best_f1 * 100.0, 2),
        'val_auc': round(val_auc, 4),
        'best_tau': round(best_tau, 3),
        'cov_lambda': cov_lambda,
        'lr_cls': optimizer_cls.param_groups[0]['lr'],
        'lr_gen': optimizer_gen.param_groups[0]['lr'],
        'epoch_sec': round(elapsed, 1),
        'patience': early_stopping.counter
    }
    epoch_history.append(epoch_record)
    pd.DataFrame(epoch_history).to_csv(TRAIN_RESULTS_CSV, index=False)
    
    epoch_pbar.set_postfix({
        'tr_loss': f"{avg_tr_loss:.3f}",
        'va_loss': f"{avg_va_loss:.3f}",
        'F1%': f"{best_f1*100:.1f}",
        'AUC': f"{val_auc:.3f}",
        'pat': f"{early_stopping.counter}/{PATIENCE}"
    })
    
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} [{elapsed:.1f}s] "
          f"TrLoss: {avg_tr_loss:.4f} (Cls: {avg_tr_cls:.4f}, Gen: {avg_tr_gen:.4f}) | "
          f"VaLoss: {avg_va_loss:.4f} (Cls: {avg_va_cls:.4f}, Gen: {avg_va_gen:.4f}) | "
          f"Macro-F1: {best_f1*100:.2f}% (tau* = {best_tau:.2f}) | AUC: {val_auc:.4f} | PPL: {val_ppl:.1f} {trophy}")
          
    if early_stopping.early_stop:
        print(f"🛑 Early stopping triggered after {PATIENCE} epochs of no improvement.")
        break

print(f"✓ Decoupled Training complete! Best Checkpoint saved at: '{CKPT_BEST}'")

# --- Cell 11 (starts at line 1265) ---
# Restore best checkpoint metadata
best_ckpt = torch.load(CKPT_BEST, map_location=device, weights_only=False)
raw_model.load_state_dict(best_ckpt['model_state_dict'])
GLOBAL_TAU = best_ckpt.get('best_tau', 0.50)
print(f"✓ Successfully loaded best model from Epoch {best_ckpt.get('epoch', '?')} (Optimal Threshold tau* = {GLOBAL_TAU:.2f})")

# Read historical telemetry
df_train = pd.read_csv(TRAIN_RESULTS_CSV)

fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=300)

# Panel 1: Multi-Task Losses (Cls BCE vs Gen NLL)
axes[0].plot(df_train['epoch'], df_train['train_cls_loss'], label='Train Cls (BCE)', color='#1f77b4', lw=2.0)
axes[0].plot(df_train['epoch'], df_train['val_cls_loss'], label='Val Cls (BCE)', color='#2ca02c', lw=2.0, linestyle='--')
axes[0].plot(df_train['epoch'], df_train['train_gen_loss'], label='Train Gen (NLL)', color='#ff7f0e', lw=1.8, linestyle=':')
axes[0].plot(df_train['epoch'], df_train['val_gen_loss'], label='Val Gen (NLL)', color='#d62728', lw=1.8, linestyle='-.')
axes[0].set_title('Decoupled Training & Validation Losses', fontsize=11, fontweight='bold', pad=10)
axes[0].set_xlabel('Epoch', fontsize=10)
axes[0].set_ylabel('Loss', fontsize=10)
axes[0].legend(fontsize=8)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Panel 2: Classifier Discriminative Ability (ROC-AUC & Accuracy)
axes[1].plot(df_train['epoch'], df_train['val_auc'], label='Val ROC-AUC', color='#9467bd', lw=2.2)
axes[1].plot(df_train['epoch'], df_train['val_acc'] / 100.0, label='Val Accuracy', color='#8c564b', lw=1.8, linestyle='--')
axes[1].set_title('Classifier Discriminative Power', fontsize=11, fontweight='bold', pad=10)
axes[1].set_xlabel('Epoch', fontsize=10)
axes[1].set_ylabel('Score', fontsize=10)
axes[1].legend(fontsize=8)
axes[1].grid(True, linestyle=':', alpha=0.6)

# Panel 3: Headline Generation Perplexity & Macro-F1
ax = axes[2]
ax_ppl = ax.twinx()
l1 = ax.plot(df_train['epoch'], df_train['val_macro_f1'], label='Classifier Macro-F1 (%)', color='#e377c2', lw=2.2, marker='o', markersize=3)
l2 = ax_ppl.plot(df_train['epoch'], df_train['val_ppl'], label='Generator PPL', color='#17becf', lw=2.0, linestyle='--', marker='x', markersize=3)
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax.set_title('Macro-F1 & Perplexity', fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Macro-F1 (%)', fontsize=10, color='#e377c2')
ax_ppl.set_ylabel('Perplexity (PPL)', fontsize=10, color='#17becf')
ax.legend(lines, labels, loc='center left', fontsize=8)
ax.grid(True, linestyle=':', alpha=0.6)

# Panel 4: Dual Learning Rate Schedules (Decoupled Optimizers)
if 'lr_cls' in df_train.columns and 'lr_gen' in df_train.columns:
    axes[3].plot(df_train['epoch'], df_train['lr_cls'], label='Head 1 Cls (AdamW 2e-4)', color='#2ca02c', lw=2.0)
    axes[3].plot(df_train['epoch'], df_train['lr_gen'], label='Head 2 Gen (AdamW 5e-4)', color='#d62728', lw=2.0, linestyle='--')
elif 'learning_rate' in df_train.columns:
    axes[3].plot(df_train['epoch'], df_train['learning_rate'], label='Optimizer LR', color='#1f77b4', lw=2.0)
axes[3].set_title('Dual Decoupled LR Schedules', fontsize=11, fontweight='bold', pad=10)
axes[3].set_xlabel('Epoch', fontsize=10)
axes[3].set_ylabel('Learning Rate', fontsize=10)
axes[3].set_yscale('log')
axes[3].legend(fontsize=8)
axes[3].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
TRAIN_CURVES_PNG = f"{OUT}/training_curves.png"
plt.savefig(TRAIN_CURVES_PNG, bbox_inches='tight', dpi=300)
plt.show()
print(f"✓ Publication training curves saved to: '{TRAIN_CURVES_PNG}'")

# --- Cell 12 (starts at line 1330) ---
class BeamSearchHypothesis:
    def __init__(
        self,
        tokens: List[int],
        log_prob: float,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        coverage: torch.Tensor
    ):
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state
        self.context = context
        self.coverage = coverage

    def get_score(self, alpha: float = 0.7, beta: float = 0.0) -> float:
        lp = math.pow((5.0 + len(self.tokens)) / 6.0, alpha)
        cov_pen = 0.0
        if beta > 0.0 and self.coverage is not None:
            cov_pen = torch.sum(torch.clamp(self.coverage - 1.0, min=0.0)).item() * beta
        return (self.log_prob / max(lp, 1e-6)) - cov_pen

    @property
    def score(self) -> float:
        return self.get_score(0.7, 0.0)


@torch.no_grad()
def generate_declickbaited_headline_beam(
    model: nn.Module,
    content_ids: torch.Tensor,
    content_mask: Optional[torch.Tensor] = None,
    src_extended_ids: Optional[torch.Tensor] = None,
    max_oovs: int = 0,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    pad_id: int = PAD_ID,
    beam_width: int = 4,
    max_len: int = 24,
    min_len: int = 4,
    alpha: float = 0.7,
    beta: float = 0.0,
    oov_tokens_map: Optional[Dict[int, str]] = None,
    tokenizer: Optional[Any] = None
) -> Union[List[int], str]:
    """
    Constrained beam search generation using Head 2 BiLSTM encoder + Coverage decoder.
    Implements trigram repetition blocking, length penalty, and extended dynamic OOV copying.
    """
    raw_m = model.module if isinstance(model, nn.DataParallel) else model
    raw_m.eval()
    dev = content_ids.device
    B = content_ids.size(0)

    if content_mask is None:
        content_mask = (content_ids != pad_id).long()

    # Step 1: Encode lead content via Head 2 BiLSTM
    c_emb = raw_m.shared_embedding(content_ids)
    enc_outputs, (enc_h, enc_c) = raw_m.enc_lstm(c_emb)
    h_0, c_0 = raw_m.decoder_head.init_decoder_state(enc_h, enc_c)
    hidden_dim = getattr(model, "hidden_dim", enc_h.size(-1))
    context_0 = torch.zeros(B, hidden_dim * 2, device=dev)
    cov_0 = torch.zeros(B, content_ids.size(1), device=dev)

    root = BeamSearchHypothesis(
        tokens=[bos_id],
        log_prob=0.0,
        state=(h_0, c_0),
        context=context_0,
        coverage=cov_0
    )
    beams: List[BeamSearchHypothesis] = [root]
    completed: List[BeamSearchHypothesis] = []

    for step in range(max_len):
        candidates: List[BeamSearchHypothesis] = []
        for hyp in beams:
            last_tok = hyp.tokens[-1]
            if last_tok == eos_id and len(hyp.tokens) > 1:
                completed.append(hyp)
                continue

            last_clamped = min(last_tok, raw_m.vocab_size - 1)
            emb = raw_m.shared_embedding(torch.tensor([last_clamped], device=dev))
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)

            final_dist, next_state, next_ctx, attn, next_cov, _ = raw_m.decoder_head.forward_step(
                input_emb=emb,
                state=hyp.state,
                context=hyp.context,
                encoder_outputs=enc_outputs,
                coverage=hyp.coverage,
                enc_mask=content_mask,
                src_extended_ids=src_extended_ids,
                max_oovs=max_oovs
            )

            if len(hyp.tokens) - 1 < min_len:
                final_dist[0, eos_id] = 1e-12
            if pad_id < final_dist.size(1):
                final_dist[0, pad_id] = 1e-12

            # Trigram repetition blocking
            if len(hyp.tokens) >= 3:
                tri = tuple(hyp.tokens[-2:])
                for past_idx in range(len(hyp.tokens) - 2):
                    if tuple(hyp.tokens[past_idx:past_idx + 2]) == tri:
                        forbidden = hyp.tokens[past_idx + 2]
                        if forbidden < final_dist.size(1):
                            final_dist[0, forbidden] = 1e-12

            log_probs = torch.log(final_dist.clamp_min(1e-12)).squeeze(0)
            topk_k = min(beam_width * 2, log_probs.size(0))
            topk_probs, topk_ids = torch.topk(log_probs, k=topk_k)

            for lp, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
                candidates.append(BeamSearchHypothesis(
                    tokens=hyp.tokens + [tid],
                    log_prob=hyp.log_prob + lp,
                    state=next_state,
                    context=next_ctx,
                    coverage=next_cov
                ))

        if not candidates:
            break
        candidates.sort(key=lambda h: h.get_score(alpha, beta), reverse=True)
        beams = candidates[:beam_width]
        if len(completed) >= beam_width:
            break

    all_hyp = completed if completed else beams
    best_hyp = max(all_hyp, key=lambda h: h.get_score(alpha, beta))
    out_ids = [t for t in best_hyp.tokens if t not in (bos_id, eos_id, pad_id)]

    if tokenizer is None:
        return out_ids

    if oov_tokens_map:
        words = []
        subwords_to_decode = []
        for tid in out_ids:
            if tid in oov_tokens_map:
                if subwords_to_decode:
                    words.append(tokenizer.decode(subwords_to_decode).replace("##", "").strip())
                    subwords_to_decode = []
                words.append(oov_tokens_map[tid].replace("##", "").strip())
            else:
                subwords_to_decode.append(tid)
        if subwords_to_decode:
            words.append(tokenizer.decode(subwords_to_decode).replace("##", "").strip())
        return " ".join([w for w in words if w]).replace("  ", " ").strip()

    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(out_ids)
    return out_ids


def autonomous_declickbait_pipeline(title_text: str, content_text: str, tau: float = GLOBAL_TAU) -> Dict[str, Any]:
    """
    End-to-End Autonomous Pipeline:
    1. Head 1 classifies input headline against article content.
    2. If clickbait (P >= tau*): generates factual replacement headline via Head 2 beam search.
    3. If genuine (P < tau*): returns original headline intact (identity mapping).
    """
    model.eval()
    t_ids = tokenizer.encode(title_text, add_special_tokens=False)[:MAX_TITLE_LEN]
    t_tensor = torch.tensor([t_ids], dtype=torch.long, device=device)
    t_mask = torch.ones_like(t_tensor)

    # Process lead content with extended dynamic OOV vocabulary
    pair = vocab_processor.process_pair(content_text, "")
    c_tokens = pair['src_tokens'][:MAX_CONTENT_LEN]
    c_ext_tokens = pair['src_extended'][:MAX_CONTENT_LEN]
    max_oovs = pair['num_oovs']
    
    c_tensor = torch.tensor([c_tokens], dtype=torch.long, device=device)
    c_mask = torch.ones_like(c_tensor)
    src_ext = torch.tensor([c_ext_tokens], dtype=torch.long, device=device)

    # Build OOV vocabulary map for exact word reconstruction
    raw_ids = tokenizer.encode(content_text, add_special_tokens=False)
    oov_tokens_map = {}
    curr_oov = 0
    for tid in raw_ids:
        if tid >= VOCAB_SIZE or tid == UNK_ID:
            if curr_oov not in oov_tokens_map:
                oov_tokens_map[VOCAB_SIZE + curr_oov] = tokenizer.decode([tid])
                curr_oov += 1

    with torch.no_grad():
        out = model(t_tensor, t_mask=t_mask, c_ids=c_tensor, c_mask=c_mask, mode="classifier", return_diagnostics=True)
        prob = out["probs"].item() if out["probs"].dim() == 0 else out["probs"][0].item()
        is_clickbait = bool(prob >= tau)

        if is_clickbait:
            action = "REMEDIATED_CLICKBAIT"
            final_headline = generate_declickbaited_headline_beam(
                model=model,
                content_ids=c_tensor,
                content_mask=c_mask,
                src_extended_ids=src_ext,
                max_oovs=max_oovs,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                pad_id=PAD_ID,
                beam_width=4,
                max_len=24,
                min_len=4,
                oov_tokens_map=oov_tokens_map,
                tokenizer=tokenizer
            )
            if isinstance(final_headline, list):
                final_headline = tokenizer.decode(final_headline)
        else:
            action = "PASSTHROUGH_GENUINE"
            final_headline = title_text

    cos_ev = out.get("cos_evidence", torch.tensor(0.0))
    cos_val = cos_ev.item() if cos_ev.dim() == 0 else cos_ev[0].item()
    min_tok = out.get("min_token_cos", torch.tensor(0.0))
    min_val = min_tok.item() if min_tok.dim() == 0 else min_tok[0].item()

    return {
        "original_headline": title_text,
        "final_headline": final_headline,
        "is_clickbait": is_clickbait,
        "clickbait_probability": round(prob, 4),
        "decision_threshold": tau,
        "action_taken": action,
        "evidence_cosine": round(cos_val, 4),
        "unsupported_bait_detector": round(min_val, 4)
    }

print("✓ Autonomous Declickbait Pipeline initialized!")

# --- Cell 13 (starts at line 1569) ---
from rouge_score import rouge_scorer

class BengaliRougeTokenizer:
    def __init__(self):
        self.pattern = _re.compile(r'[ঀ-৿a-zA-Z0-9]+')
    def tokenize(self, text: str):
        return self.pattern.findall(unicodedata.normalize('NFKC', str(text)).lower())

bengali_tok = BengaliRougeTokenizer()
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=bengali_tok)

RESULT_CSV = f"{OUT}/result.csv"
print(f"📊 Evaluating Held-Out Test Set ({len(test_dataset):,d} instances)...\n")

eval_samples = test_dataset.df.to_dict('records')[:300]
results_table = []
lang_metrics = defaultdict(lambda: {
    'cls_acc': [], 'cls_f1': [],
    'r1_f': [], 'r1_p': [], 'r1_r': [],
    'r2_f': [], 'r2_p': [], 'r2_r': [],
    'rl_f': [], 'rl_p': [], 'rl_r': [],
    'ref_len': [], 'gen_len': []
})

pbar = tqdm(eval_samples, desc="Evaluating Multi-Head Test Samples", dynamic_ncols=True, unit="sample")

for i, row in enumerate(pbar, start=1):
    orig_title = row['title']
    content_body = row['content']
    ref_title = row['target_title']
    true_label = int(row['label'])
    lang = row['language']
    
    pipeline_res = autonomous_declickbait_pipeline(orig_title, content_body, tau=GLOBAL_TAU)
    pred_label = 1 if pipeline_res['is_clickbait'] else 0
    final_headline = pipeline_res['final_headline']
    
    # ROUGE scoring against target reference headline
    scores = scorer.score(ref_title, final_headline)
    r1_f, r1_p, r1_r = scores['rouge1'].fmeasure, scores['rouge1'].precision, scores['rouge1'].recall
    r2_f, r2_p, r2_r = scores['rouge2'].fmeasure, scores['rouge2'].precision, scores['rouge2'].recall
    rl_f, rl_p, rl_r = scores['rougeL'].fmeasure, scores['rougeL'].precision, scores['rougeL'].recall
    
    ref_toks = len(bengali_tok.tokenize(ref_title))
    gen_toks = len(bengali_tok.tokenize(final_headline))
    is_correct = int(pred_label == true_label)
    
    for scope in (lang, 'overall'):
        lang_metrics[scope]['cls_acc'].append(is_correct)
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
        'original_headline': orig_title,
        'reference_headline': ref_title,
        'final_headline': final_headline,
        'true_label': true_label,
        'pred_label': pred_label,
        'is_clickbait': pipeline_res['is_clickbait'],
        'clickbait_prob': pipeline_res['clickbait_probability'],
        'action_taken': pipeline_res['action_taken'],
        'evidence_cosine': pipeline_res['evidence_cosine'],
        'unsupported_bait_detector': pipeline_res['unsupported_bait_detector'],
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
        'ClsAcc': f"{np.mean(lang_metrics['overall']['cls_acc'])*100:.1f}%",
        'R1-F1': f"{np.mean(lang_metrics['overall']['r1_f']):.3f}",
        'RL-F1': f"{np.mean(lang_metrics['overall']['rl_f']):.3f}"
    })

df_results = pd.DataFrame(results_table)
df_results.to_csv(RESULT_CSV, index=False)
print(f"\n✓ Saved full test prediction records to: '{RESULT_CSV}'")

print("\n" + "=" * 94)
print("🏆 FINAL COMPREHENSIVE TEST EVALUATION METRICS (Option 1 Decoupled 2x8 Multi-Head System 11)")
print("=" * 94)
print(f"{'Language':<12} | {'N':<4} | {'Cls Acc':<8} | {'ROUGE-1 (F1/P/R)':<22} | {'ROUGE-2 (F1/P/R)':<22} | {'ROUGE-L (F1/P/R)':<22}")
print("-" * 94)
for lg, mets in sorted(lang_metrics.items()):
    n = len(mets['r1_f'])
    acc_str = f"{np.mean(mets['cls_acc'])*100:.1f}%"
    r1_str = f"{np.mean(mets['r1_f']):.3f} / {np.mean(mets['r1_p']):.3f} / {np.mean(mets['r1_r']):.3f}"
    r2_str = f"{np.mean(mets['r2_f']):.3f} / {np.mean(mets['r2_p']):.3f} / {np.mean(mets['r2_r']):.3f}"
    rl_str = f"{np.mean(mets['rl_f']):.3f} / {np.mean(mets['rl_p']):.3f} / {np.mean(mets['rl_r']):.3f}"
    print(f"{lg:<12} | {n:<4} | {acc_str:<8} | {r1_str:<22} | {r2_str:<22} | {rl_str:<22}")
print("=" * 94)

# --- Cell 14 (starts at line 1681) ---
CLASSIFIER_DIAG_PNG = f"{OUT}/classifier_diagnostics.png"
ROUGE_METRICS_PNG = f"{OUT}/rouge_metrics.png"
MULTILINGUAL_PNG = f"{OUT}/multilingual_breakdown.png"

if os.path.exists(RESULT_CSV):
    df_eval = pd.read_csv(RESULT_CSV)
    
    # 1. Classifier Diagnostics (ROC & PR Curves)
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    fpr, tpr, _ = roc_curve(df_eval['true_label'], df_eval['clickbait_prob'])
    prec, rec, _ = precision_recall_curve(df_eval['true_label'], df_eval['clickbait_prob'])
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prec)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.2, label=f'Head 1 ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].plot(rec, prec, color='#2ca02c', lw=2.2, label=f'Precision-Recall (AUC = {pr_auc:.3f})')
    axes[1].axvline(x=GLOBAL_TAU, color='red', linestyle='--', alpha=0.7, label=f'Optimal tau* = {GLOBAL_TAU:.2f}')
    axes[1].set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Recall', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].legend(loc='lower left', fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(CLASSIFIER_DIAG_PNG, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✓ Saved Classifier Diagnostics to '{CLASSIFIER_DIAG_PNG}'")
    
    # 2. ROUGE Distributions & Precision vs Recall
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    metric_cols = ['rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1']
    data_to_plot = [df_eval[c].values for c in metric_cols]
    box = axes[0].boxplot(data_to_plot, patch_artist=True, labels=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'], medianprops=dict(color='black', linewidth=1.5))
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for idx, d in enumerate(data_to_plot, start=1):
        m = np.mean(d)
        axes[0].scatter([idx], [m], color='red', marker='D', s=50, zorder=5)
        axes[0].annotate(f"Mean: {m:.3f}", (idx, m), textcoords="offset points", xytext=(12, -3), fontsize=9, fontweight='bold')
    axes[0].set_title("ROUGE F1 Distributions on Held-Out Test Set", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_ylabel("F1 Score", fontsize=11)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].scatter(df_eval['rouge_1_recall'], df_eval['rouge_1_precision'], alpha=0.5, color='#2ca02c', s=35, label='ROUGE-1')
    axes[1].scatter(df_eval['rouge_l_recall'], df_eval['rouge_l_precision'], alpha=0.5, color='#1f77b4', s=35, label='ROUGE-L')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Balanced P=R')
    axes[1].set_title("Headline Generation Precision vs Recall", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc='lower left', fontsize=9)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(ROUGE_METRICS_PNG, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✓ Saved ROUGE distributions to '{ROUGE_METRICS_PNG}'")
    
    # 3. Multilingual Stratified Breakdown
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    langs_present = sorted(df_eval['language'].unique())
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
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    ax.set_title("Trilingual Performance (Bengali, Banglish, English)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([lg.capitalize() for lg in langs_present], fontsize=11, fontweight='bold')
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(MULTILINGUAL_PNG, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✓ Saved Multilingual breakdown to '{MULTILINGUAL_PNG}'")

# --- Cell 15 (starts at line 1781) ---
# =====================================================================
# INTERACTIVE DEMONSTRATION OF AUTONOMOUS INFERENCE GATE
# =====================================================================

sample_articles = [
    {
        "title": "অবশেষে মুখ খুললেন তারকা! যা বললেন শুনে সবাই স্তব্ধ!",
        "content": "জাতীয় দলের তারকা অলরাউন্ডার আজ এক সংবাদ সম্মেলনে কথা বলেছেন। ইনজুরি কাটিয়ে আসন্ন সিরিজেই তিনি দলে ফিরছেন বলে নিশ্চিত করেছে বিসিবি।"
    },
    {
        "title": "বাংলাদেশ দল আজ শ্রীলঙ্কার বিপক্ষে মাঠে নামছে",
        "content": "আজ দুপুর আড়াইটায় এশিয়া কাপের গুরুত্বপূর্ণ ম্যাচে শ্রীলঙ্কার মুখোমুখি হবে বাংলাদেশ জাতীয় ক্রিকেট দল। ম্যাচটি সরাসরি সম্প্রচার করবে টি স্পোর্টস।"
    }
]

print("=" * 80)
print("🤖 AUTONOMOUS MULTI-HEAD DECLICKBAITING IN ACTION")
print("=" * 80)

for idx, sample in enumerate(sample_articles, 1):
    res = autonomous_declickbait_pipeline(sample['title'], sample['content'], tau=GLOBAL_TAU)
    print(f"Sample #{idx}:")
    print(f"  Input Headline:       {res['original_headline']}")
    print(f"  P(Clickbait):         {res['clickbait_probability']:.4f} (Threshold tau* = {res['decision_threshold']:.2f})")
    print(f"  Verdict & Action:     {res['action_taken']}")
    print(f"  Final Headline:       {res['final_headline']}")
    print(f"  Evidence Cosine:      {res['evidence_cosine']:.4f}")
    print(f"  Unsupported Bait Sim: {res['unsupported_bait_detector']:.4f} (Min token cosine)")
    print("-" * 80)

# --- Cell 16 (starts at line 1812) ---
import glob, os, shutil, zipfile
from pathlib import Path

PACKAGE_ZIP = "result.zip"
ALIAS_ZIP = "10_multihead_bpe_pointer_generator_package.zip"

files_to_pack = [
    CKPT_BEST,
    CKPT_LAST,
    RESULT_CSV,
    TRAIN_RESULTS_CSV,
    f"{OUT}/training_curves.png",
    f"{OUT}/classifier_diagnostics.png",
    f"{OUT}/rouge_metrics.png",
    f"{OUT}/multilingual_breakdown.png"
]

existing_files = [f for f in files_to_pack if os.path.exists(f)]

print(f"📦 Packaging {len(existing_files)} lean artifact files into '{PACKAGE_ZIP}'...")
with zipfile.ZipFile(PACKAGE_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for f in existing_files:
        arcname = os.path.basename(f)
        zf.write(f, arcname=arcname)
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  + Added: {arcname:<32} ({size_mb:.2f} MB)")

shutil.copy2(PACKAGE_ZIP, ALIAS_ZIP)

zip_size_mb = Path(PACKAGE_ZIP).stat().st_size / (1024 * 1024) if Path(PACKAGE_ZIP).exists() else 0.0
print("=" * 70)
print(f"📦 LEAN MULTI-HEAD PACKAGE READY: {PACKAGE_ZIP} ({zip_size_mb:.2f} MB)")
print("  Archive Contents (Epoch .pt files strictly omitted):")
for f in existing_files:
    print(f"   - {os.path.basename(f)}")
print("=" * 70)

try:
    from google.colab import files
    files.download(PACKAGE_ZIP)
    print("✓ Triggered Colab direct download for result.zip!")
except Exception:
    print(f"✓ Output package ready at: {os.path.abspath(PACKAGE_ZIP)}")


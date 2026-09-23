# --- Cell 3 (starts at line 1) ---
# Install dependencies for tokenizer, training, evaluation, and ONNX runtime
!pip install -q tokenizers transformers datasets scikit-learn pandas matplotlib seaborn onnx onnxruntime tqdm
print("✓ Core dependencies installed successfully!")

# --- Cell 5 (starts at line 6) ---
import os
from pathlib import Path

# Mount Google Drive if on Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/declickbait'
    os.makedirs(f"{DRIVE_ROOT}/outputs_02v3", exist_ok=True)
    OUT = f"{DRIVE_ROOT}/outputs_02v3"
    print(f"✓ Google Drive mounted! Outputs will persist to: {OUT}")
except Exception:
    OUT = "./outputs_02v3"
    os.makedirs(OUT, exist_ok=True)
    print(f"ℹ️ Running in standard environment. Outputs will save to: {OUT}")

# --- Cell 7 (starts at line 23) ---
import random
import numpy as np
import torch
import torch.nn as nn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = max(torch.cuda.device_count(), 1)
print(f"🌍 Compute Device: {device} | Active GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for g in range(torch.cuda.device_count()):
        print(f"   GPU {g}: {torch.cuda.get_device_name(g)} ({torch.cuda.get_device_properties(g).total_memory / 1e9:.1f} GB)")

def print_mem(tag: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"   [{tag}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# --- Cell 9 (starts at line 51) ---
import glob
import json
import pandas as pd

def find_splits():
    candidates = [
        "splits/train_train_val_test.jsonl",
        "../splits/train_train_val_test.jsonl",
        "/content/drive/MyDrive/declickbait/splits/train_train_val_test.jsonl",
        "/content/splits/train_train_val_test.jsonl",
        "/kaggle/input/declickbait/splits/train_train_val_test.jsonl",
        "/kaggle/input/*/splits/train_train_val_test.jsonl",
        "data/prepared/corpus_balanced.jsonl"
    ]
    for c in candidates:
        matches = glob.glob(c)
        if matches and os.path.exists(matches[0]):
            split_dir = os.path.dirname(matches[0])
            print(f"✓ Found verified data at: {split_dir}")
            return split_dir
    return None

SPLIT_DIR = find_splits()

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)

if SPLIT_DIR and os.path.exists(f"{SPLIT_DIR}/train_train_val_test.jsonl"):
    df_train = load_jsonl(f"{SPLIT_DIR}/train_train_val_test.jsonl")
    df_val = load_jsonl(f"{SPLIT_DIR}/val_train_val_test.jsonl")
    df_test = load_jsonl(f"{SPLIT_DIR}/test_train_val_test.jsonl")
else:
    # Auto-extract from zip or fallback to prepared corpus
    zip_path = "declickbait-splits.zip"
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(".")
        SPLIT_DIR = "splits"
        df_train = load_jsonl(f"{SPLIT_DIR}/train_train_val_test.jsonl")
        df_val = load_jsonl(f"{SPLIT_DIR}/val_train_val_test.jsonl")
        df_test = load_jsonl(f"{SPLIT_DIR}/test_train_val_test.jsonl")
    else:
        raise FileNotFoundError("Could not locate dataset splits. Please place splits/ or declickbait-splits.zip in workspace.")

print(f"✓ Dataset Loaded: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")
print("Sample Columns:", df_train.columns.tolist())

# --- Cell 11 (starts at line 106) ---
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

VOCAB_SIZE = 16000
TOKENIZER_PATH = f"{OUT}/byte_bpe_16k.json"

if os.path.exists(TOKENIZER_PATH):
    print(f"Loading existing Byte-Level BPE tokenizer from: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
else:
    print(f"Training fresh Byte-Level BPE Tokenizer (Vocab Size: {VOCAB_SIZE})...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        min_frequency=2,
        show_progress=True
    )

    # Gather training texts across headlines and body context
    train_texts = (
        df_train["original_title"].dropna().astype(str).tolist() +
        df_train["body"].dropna().astype(str).tolist()
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    tokenizer.save(TOKENIZER_PATH)
    print(f"✓ Trained & Saved Byte-Level BPE Tokenizer to: {TOKENIZER_PATH}")

# Ensure no hard truncation cap is stored in tokenizer object
tokenizer.no_truncation()

PAD_ID = tokenizer.token_to_id("[PAD]")
UNK_ID = tokenizer.token_to_id("[UNK]")
print(f"✓ Tokenizer Ready: Vocab Size = {tokenizer.get_vocab_size()} | PAD_ID = {PAD_ID}")

# --- Cell 13 (starts at line 148) ---
from torch.utils.data import Dataset, DataLoader

MAX_TITLE_LEN = 64
MAX_CONTENT_LEN = 768
BATCH_SIZE = 32 * N_GPUS  # 64 on Dual T4, 32 on Single T4

class FastBPEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer, max_t_len: int = 64, max_c_len: int = 768):
        self.titles = df["original_title"].fillna("").astype(str).tolist()
        self.contents = df["body"].fillna("").astype(str).tolist()
        self.labels = df["clickbait"].astype(float).tolist()
        self.languages = df.get("language_mix", ["unknown"] * len(df)).tolist()
        self.tokenizer = tokenizer
        self.max_t_len = max_t_len
        self.max_c_len = max_c_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        # Encode Title
        t_enc = self.tokenizer.encode(self.titles[idx])
        t_ids = t_enc.ids[:self.max_t_len]
        t_mask = [1] * len(t_ids)
        if len(t_ids) < self.max_t_len:
            pad_n = self.max_t_len - len(t_ids)
            t_ids += [PAD_ID] * pad_n
            t_mask += [0] * pad_n

        # Encode Content / Body Context
        c_enc = self.tokenizer.encode(self.contents[idx])
        c_ids = c_enc.ids[:self.max_c_len]
        c_mask = [1] * len(c_ids)
        if len(c_ids) < self.max_c_len:
            pad_n = self.max_c_len - len(c_ids)
            c_ids += [PAD_ID] * pad_n
            c_mask += [0] * pad_n

        return {
            "t_ids": torch.tensor(t_ids, dtype=torch.long),
            "t_mask": torch.tensor(t_mask, dtype=torch.long),
            "c_ids": torch.tensor(c_ids, dtype=torch.long),
            "c_mask": torch.tensor(c_mask, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
            "lang": self.languages[idx]
        }

train_ds = FastBPEDataset(df_train, tokenizer, MAX_TITLE_LEN, MAX_CONTENT_LEN)
val_ds = FastBPEDataset(df_val, tokenizer, MAX_TITLE_LEN, MAX_CONTENT_LEN)
test_ds = FastBPEDataset(df_test, tokenizer, MAX_TITLE_LEN, MAX_CONTENT_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

print(f"✓ DataLoaders ready: Batch Size={BATCH_SIZE} | {len(train_loader)} Train Batches | {len(val_loader)} Val Batches")

# --- Cell 15 (starts at line 206) ---
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerTowerBlock(nn.Module):
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

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoderTower(nn.Module):
    def __init__(self, num_layers: int, d_model: int = 128, n_heads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerTowerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask=None):
        key_padding_mask = ~mask.bool() if mask is not None else None
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.final_norm(x)


class TitleGuidedEvidenceCrossAttention(nn.Module):
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

    def forward(self, t_emb, t_mask, c_emb, c_mask, return_diagnostics: bool = False):
        key_padding_mask = ~c_mask.bool() if c_mask is not None else None
        a_ev, attn_weights = self.cross_attn(
            query=t_emb, key=c_emb, value=c_emb,
            key_padding_mask=key_padding_mask,
            need_weights=return_diagnostics
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

        feat = torch.cat([
            h_title, h_evidence, h_content_global,
            diff_evidence, prod_evidence,
            cos_evidence, cos_global, min_tok_cos
        ], dim=-1)

        fused = self.fuse(feat)
        diagnostics = {
            "cos_evidence": cos_evidence.squeeze(-1),
            "cos_global": cos_global.squeeze(-1),
            "min_token_cos": min_tok_cos.squeeze(-1),
            "tok_cos_per_token": tok_cos_masked,
            "attn_weights": attn_weights if return_diagnostics else None
        }
        return fused, diagnostics


class BPEEvidenceClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 128,
        n_heads: int = 4,
        title_layers: int = 3,
        content_layers: int = 4,
        d_ff: int = 512,
        max_title_len: int = 64,
        max_content_len: int = 768,
        pad_idx: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(max_title_len, max_content_len) + 16, dropout=dropout)

        self.title_tower = TransformerEncoderTower(title_layers, d_model, n_heads, d_ff, dropout)
        self.content_tower = TransformerEncoderTower(content_layers, d_model, n_heads, d_ff, dropout)
        self.evidence_aligner = TitleGuidedEvidenceCrossAttention(d_model, n_heads, dropout)

        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
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

    def forward(self, t_ids, t_mask, c_ids, c_mask, labels=None, return_diagnostics: bool = False):
        t_repr = self.title_tower(self.pos_encoder(self.shared_embedding(t_ids)), mask=t_mask)
        c_repr = self.content_tower(self.pos_encoder(self.shared_embedding(c_ids)), mask=c_mask)

        fused, diag = self.evidence_aligner(t_repr, t_mask, c_repr, c_mask, return_diagnostics=return_diagnostics)
        logits = self.classifier_head(fused).squeeze(-1)
        probs = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return {
            "logits": logits,
            "probs": probs,
            "loss": loss,
            "cos_evidence": diag["cos_evidence"],
            "cos_global": diag["cos_global"],
            "min_token_cos": diag["min_token_cos"]
        }

raw_model = BPEEvidenceClassifier(
    vocab_size=VOCAB_SIZE,
    d_model=128,
    n_heads=4,
    title_layers=3,
    content_layers=4,
    d_ff=512,
    max_title_len=MAX_TITLE_LEN,
    max_content_len=MAX_CONTENT_LEN,
    pad_idx=PAD_ID
).to(device)

if torch.cuda.device_count() > 1:
    print(f"⚡ Kaggle Dual T4 detected! Wrapping in DataParallel across {torch.cuda.device_count()} GPUs.")
    model_train = nn.DataParallel(raw_model)
else:
    model_train = raw_model

total_params = sum(p.numel() for p in raw_model.parameters())
trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
print(f"✓ Model Loaded: Total Parameters = {total_params / 1e6:.2f}M | Trainable = {trainable_params / 1e6:.2f}M")
print_mem("Model Ready")

# --- Cell 17 (starts at line 404) ---
import copy

EPOCHS = 50
PATIENCE = 10
CHECKPOINT_DIR = f"{OUT}/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CKPT_BEST = f"{CHECKPOINT_DIR}/checkpoint_best.pt"
CKPT_LAST = f"{CHECKPOINT_DIR}/checkpoint_last.pt"
TRAIN_RESULTS_CSV = f"{OUT}/training_results.csv"

# Differential learning rates: smaller for deep encoder towers, larger for fusion & classifier
tower_params = (
    list(raw_model.shared_embedding.parameters()) +
    list(raw_model.title_tower.parameters()) +
    list(raw_model.content_tower.parameters())
)
head_params = (
    list(raw_model.evidence_aligner.parameters()) +
    list(raw_model.classifier_head.parameters())
)

optimizer = torch.optim.AdamW([
    {"params": tower_params, "lr": 1e-4, "weight_decay": 1e-4},
    {"params": head_params, "lr": 3e-4, "weight_decay": 1e-5}
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS * len(train_loader),
    eta_min=1e-6
)

start_epoch = 1
epoch_history = []
best_val_f1 = -1.0
patience_counter = 0

# Crash-Proof Resumption Check
if os.path.exists(CKPT_LAST):
    print(f"🔄 Resumption checkpoint found at: {CKPT_LAST}")
    ckpt = torch.load(CKPT_LAST, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val_f1 = ckpt.get("best_val_f1", 0.0)
    patience_counter = ckpt.get("patience_counter", 0)
    epoch_history = ckpt.get("epoch_history", [])
    print(f"✓ Successfully resumed from Epoch {ckpt['epoch']} (Best Val Macro-F1: {best_val_f1*100:.2f}%)")
else:
    print("🚀 Fresh training run initialized.")

# --- Cell 19 (starts at line 458) ---
import time
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

print(f"🚀 Launching Training: Epochs {start_epoch} to {EPOCHS} (Early Stopping Patience: {PATIENCE})...")

epoch_pbar = tqdm(range(start_epoch, EPOCHS + 1), desc="Overall Training Progress", unit="epoch")

for epoch in epoch_pbar:
    t0 = time.time()
    model_train.train()
    total_train_loss = 0.0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{EPOCHS:2d} [Train]", leave=False, unit="batch")
    for step, batch in enumerate(train_pbar):
        t_ids = batch["t_ids"].to(device)
        t_mask = batch["t_mask"].to(device)
        c_ids = batch["c_ids"].to(device)
        c_mask = batch["c_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        out = model_train(t_ids=t_ids, t_mask=t_mask, c_ids=c_ids, c_mask=c_mask, labels=labels)
        loss = out["loss"].mean() if out["loss"].ndim > 0 else out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_train_loss / (step + 1):.4f}"})

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation Phase
    model_train.eval()
    val_loss = 0.0
    val_probs, val_labels = [], []

    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:2d}/{EPOCHS:2d} [Val]", leave=False, unit="batch")
    with torch.no_grad():
        for batch in val_pbar:
            t_ids = batch["t_ids"].to(device)
            t_mask = batch["t_mask"].to(device)
            c_ids = batch["c_ids"].to(device)
            c_mask = batch["c_mask"].to(device)
            labels = batch["label"].to(device)

            out = model_train(t_ids=t_ids, t_mask=t_mask, c_ids=c_ids, c_mask=c_mask, labels=labels)
            loss = out["loss"].mean() if out["loss"].ndim > 0 else out["loss"]
            val_loss += loss.item()

            val_probs.extend(out["probs"].cpu().tolist())
            val_labels.extend(labels.cpu().tolist())
            val_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    avg_val_loss = val_loss / len(val_loader)
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

    # Threshold calibration (optimal tau* maximizing Macro-F1)
    best_tau = 0.50
    best_tau_f1 = 0.0
    for tau in np.linspace(0.15, 0.85, 29):
        preds = (val_probs >= tau).astype(int)
        f1_cand = f1_score(val_labels, preds, average="macro", zero_division=0)
        if f1_cand > best_tau_f1:
            best_tau_f1 = f1_cand
            best_tau = tau

    val_acc = accuracy_score(val_labels, (val_probs >= best_tau).astype(int))
    val_auc = roc_auc_score(val_labels, val_probs)
    elapsed = time.time() - t0

    # Log telemetry
    epoch_entry = {
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": best_tau_f1,
        "roc_auc": val_auc,
        "optimal_tau": best_tau,
        "patience": patience_counter,
        "time_sec": elapsed
    }
    epoch_history.append(epoch_entry)
    pd.DataFrame(epoch_history).to_csv(TRAIN_RESULTS_CSV, index=False)

    print(f"Epoch {epoch:2d}/{EPOCHS:2d} ({elapsed:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {best_tau_f1*100:.2f}% (tau*={best_tau:.2f}) | ROC-AUC: {val_auc:.4f}")

    # Checkpoint Last
    last_ckpt_data = {
        "epoch": epoch,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
        "patience_counter": patience_counter,
        "best_tau": best_tau,
        "epoch_history": epoch_history
    }
    torch.save(last_ckpt_data, CKPT_LAST)

    # Early stopping check on Macro-F1
    if best_tau_f1 > best_val_f1:
        best_val_f1 = best_tau_f1
        patience_counter = 0
        best_ckpt_data = copy.deepcopy(last_ckpt_data)
        torch.save(best_ckpt_data, CKPT_BEST)
        print(f"   🏆 NEW BEST MODEL SAVED! Macro-F1: {best_val_f1*100:.2f}% -> {CKPT_BEST}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping triggered at Epoch {epoch}! No improvement in {PATIENCE} epochs.")
            break

    epoch_pbar.set_postfix({
        "train_loss": f"{avg_train_loss:.4f}",
        "val_loss": f"{avg_val_loss:.4f}",
        "val_f1": f"{best_tau_f1*100:.1f}%",
        "best_f1": f"{best_val_f1*100:.1f}%",
        "patience": f"{patience_counter}/{PATIENCE}"
    })

# --- Cell 21 (starts at line 584) ---
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

print(f"📊 Loading best checkpoint from '{CKPT_BEST}' for held-out evaluation...")
best_ckpt_data = torch.load(CKPT_BEST, map_location=device, weights_only=False)
raw_model.load_state_dict(best_ckpt_data["model_state"])
opt_tau = best_ckpt_data.get("best_tau", 0.50)
print(f"✓ Successfully restored weights from Epoch {best_ckpt_data.get('epoch', '?')} (tau* = {opt_tau:.2f})")

raw_model.eval()
test_probs, test_labels, test_langs = [], [], []
test_cos_evidence, test_min_token_cos = [], []

test_pbar = tqdm(test_loader, desc="Held-Out Test Evaluation", unit="batch")
with torch.no_grad():
    for batch in test_pbar:
        t_ids = batch["t_ids"].to(device)
        t_mask = batch["t_mask"].to(device)
        c_ids = batch["c_ids"].to(device)
        c_mask = batch["c_mask"].to(device)
        labels = batch["label"].to(device)

        out = raw_model(t_ids=t_ids, t_mask=t_mask, c_ids=c_ids, c_mask=c_mask)
        test_probs.extend(out["probs"].cpu().tolist())
        test_labels.extend(labels.cpu().tolist())
        test_langs.extend(batch["lang"])
        test_cos_evidence.extend(out["cos_evidence"].cpu().tolist())
        test_min_token_cos.extend(out["min_token_cos"].cpu().tolist())

test_probs = np.array(test_probs)
test_labels = np.array(test_labels)
test_preds = (test_probs >= opt_tau).astype(int)

macro_f1 = f1_score(test_labels, test_preds, average="macro")
acc = accuracy_score(test_labels, test_preds)
auc = roc_auc_score(test_labels, test_probs)

print("=" * 70)
print("🏆 FINAL HELD-OUT TEST EVALUATION METRICS (02_v3 BPE Evidence Attention)")
print("=" * 70)
print(f"Overall Accuracy:  {acc*100:.2f}%")
print(f"Overall Macro-F1:  {macro_f1*100:.2f}%")
print(f"ROC-AUC Score:     {auc:.4f}")
print("=" * 70)

# Per-Language Breakdown
df_eval = pd.DataFrame({
    "true": test_labels,
    "pred": test_preds,
    "prob": test_probs,
    "lang": test_langs
})

print("
--- Language Performance Breakdown ---")
for lang, g in df_eval.groupby("lang"):
    l_acc = accuracy_score(g["true"], g["pred"])
    l_f1 = f1_score(g["true"], g["pred"], average="macro", zero_division=0)
    print(f"  {lang.upper():<10} (N={len(g):4d}): Accuracy={l_acc*100:.2f}% | Macro-F1={l_f1*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Factual", "Clickbait"], yticklabels=["Factual", "Clickbait"])
plt.title(f"02_v3 Classifier Confusion Matrix (tau*={opt_tau:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.show()

# --- Cell 23 (starts at line 658) ---
raw_model.eval()
bench_t_ids = torch.randint(1, VOCAB_SIZE, (30, MAX_TITLE_LEN), device=device)
bench_t_mask = torch.ones(30, MAX_TITLE_LEN, dtype=torch.long, device=device)
bench_c_ids = torch.randint(1, VOCAB_SIZE, (30, MAX_CONTENT_LEN), device=device)
bench_c_mask = torch.ones(30, MAX_CONTENT_LEN, dtype=torch.long, device=device)

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = raw_model(bench_t_ids, bench_t_mask, bench_c_ids, bench_c_mask)

# Measure
latencies = []
for _ in range(30):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = raw_model(bench_t_ids, bench_t_mask, bench_c_ids, bench_c_mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latencies.append((time.perf_counter() - t0) * 1000.0)

mean_lat = np.mean(latencies)
p95_lat = np.percentile(latencies, 95)
per_item_ms = mean_lat / 30.0

print("=" * 65)
print(f"⚡ Batch of 30 Feed Items Latency Benchmark ({device})")
print(f"Mean Batch Latency:   {mean_lat:.2f} ms")
print(f"P95 Batch Latency:    {p95_lat:.2f} ms")
print(f"Per-Item Latency:     {per_item_ms:.2f} ms / item")
print(f"SLA Compliance:       {'✅ PASSED (<25ms)' if per_item_ms < 25.0 else '⚠️ Note latency'}")
print("=" * 65)

# --- Cell 25 (starts at line 692) ---
import shutil

ONNX_PATH = f"{OUT}/bpe_evidence_classifier.onnx"
print(f"Exporting model to ONNX: {ONNX_PATH}...")

class ONNXExportWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, t_ids, t_mask, c_ids, c_mask):
        out = self.m(t_ids, t_mask, c_ids, c_mask)
        return out["logits"], out["probs"], out["cos_evidence"], out["min_token_cos"]

onnx_wrapper = ONNXExportWrapper(raw_model).eval()
dummy_t = torch.randint(1, VOCAB_SIZE, (1, MAX_TITLE_LEN), device=device)
dummy_tm = torch.ones(1, MAX_TITLE_LEN, dtype=torch.long, device=device)
dummy_c = torch.randint(1, VOCAB_SIZE, (1, MAX_CONTENT_LEN), device=device)
dummy_cm = torch.ones(1, MAX_CONTENT_LEN, dtype=torch.long, device=device)

torch.onnx.export(
    onnx_wrapper,
    (dummy_t, dummy_tm, dummy_c, dummy_cm),
    ONNX_PATH,
    input_names=["t_ids", "t_mask", "c_ids", "c_mask"],
    output_names=["logits", "probs", "cos_evidence", "min_token_cos"],
    dynamic_axes={
        "t_ids": {0: "batch_size"},
        "t_mask": {0: "batch_size"},
        "c_ids": {0: "batch_size"},
        "c_mask": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "probs": {0: "batch_size"},
        "cos_evidence": {0: "batch_size"},
        "min_token_cos": {0: "batch_size"}
    },
    opset_version=17,
    do_constant_folding=True,
    dynamo=False
)

print(f"✓ Successfully exported ONNX model ({os.path.getsize(ONNX_PATH) / 1e6:.2f} MB)")

# Save calibration metadata for Extension
calib_meta = {
    "optimal_threshold": float(opt_tau),
    "max_title_len": MAX_TITLE_LEN,
    "max_content_len": MAX_CONTENT_LEN,
    "vocab_size": VOCAB_SIZE
}
with open(f"{OUT}/calibration_config.json", "w") as f:
    json.dump(calib_meta, f, indent=2)

# Create ZIP archive
shutil.make_archive(f"{OUT}/result_02v3_classifier", "zip", OUT)
print(f"✓ Packaged result archive: {OUT}/result_02v3_classifier.zip")


# 🏛️ Architecture Flowcharts (NLP Project)

---

## 🏆 Model 4 (11_v2): Dual-GPU Decoupled 2 × 8 Multi-Head System
**Hardware**: Kaggle Dual NVIDIA T4 (2x 16GB, DataParallel) | **Scale**: ~28.85M Parameters (~115 MB)  
**Innovations**: Option A (Auto-Quench Plateau Decay) + Option C (Composite Checkpoint Synthesis) | **Batch Size**: 128 (64/GPU)

```mermaid
flowchart TD
    classDef inputStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef dpStyle fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    classDef h1Style fill:#0f3832,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef h2Style fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef optAStyle fill:#431407,stroke:#fb923c,stroke-width:2px,color:#f8fafc;
    classDef optCStyle fill:#3b0764,stroke:#c084fc,stroke-width:2px,color:#f8fafc;
    classDef gateStyle fill:#4c0519,stroke:#fb7185,stroke-width:2px,color:#f8fafc;

    subgraph Inputs["1. Unified Trilingual Ingestion (High Context)"]
        Title["Headline Title (max 64 subwords)"]:::inputStyle
        Content["Lead Content (max 768 subwords)"]:::inputStyle
        Target["Reference Truth Title (max 64 subwords)"]:::inputStyle
        BUET["BUET BanglishBERT Tokenizer (64,768 Vocab)"]:::inputStyle
        Title & Content & Target --> BUET
    end

    subgraph DataParallel["2. Kaggle Dual-GPU DataParallel (Batch Size: 128)"]
        BUET --> DP["torch.nn.DataParallel Engine"]:::dpStyle
        DP --> GPU0["GPU 0: Tesla T4 (64 samples)"]:::dpStyle
        DP --> GPU1["GPU 1: Tesla T4 (64 samples)"]:::dpStyle
        GPU0 & GPU1 --> Emb["Shared Multilingual Embedding (64,768 × 128)<br/><i>Exclusively owned by Head 1 Cls Partition</i>"]:::h1Style
    end

    subgraph Head1["3. HEAD 1: Evidence Attention Classifier (~9.83M Params)"]
        Emb --> TT["Title Tower (3 Transformer Layers, d=128)"]:::h1Style
        Emb --> CT["Content Tower (4 Transformer Layers, d=128)"]:::h1Style
        TT & CT --> CrossAttn["Title-Guided Cross-Attention (Q=T, K=C, V=C)<br/>Token-Level Bait Word Detector (min_i cos(T_i, A_ev, i))"]:::h1Style
        CrossAttn --> ClsHead["MLP Fusion Head -> P(clickbait) ∈ [0, 1]"]:::h1Style
    end

    subgraph Head2["4. HEAD 2: BiLSTM Pointer-Generator Rewriter (~19.02M Params)"]
        Emb --> BiLSTM["Content BiLSTM Encoder (hidden=256 -> 512-d)"]:::h2Style
        BiLSTM --> Bahdanau["Bahdanau Additive Coverage Attention (c^t = sum a^t')"]:::h2Style
        Bahdanau --> PGen["Pointer-Generator Soft Switch: p_gen ∈ [0, 1]"]:::h2Style
        PGen --> Scatter["Dynamic Extended OOV Scatter-Add<br/>P(w) = p_gen·P_vocab + (1-p_gen)·sum(a_i^t)"]:::h2Style
        Scatter --> GenOut["Generated Factual Replacement Headline"]:::h2Style
    end

    subgraph OptionA["5. OPTION A: Asymmetric Optimization & Dynamic Auto-Quench"]
        ClsHead --> LossCls["L_cls: Binary Cross-Entropy"]:::optAStyle
        GenOut --> LossGen["L_gen: NLL + λ · Coverage Penalty"]:::optAStyle
        LossCls --> OptCls["<b>optimizer_cls</b>: AdamW(lr=8e-5, wd=1e-3)<br/><b>ReduceLROnPlateau</b> on Val Macro-F1<br/><i>Auto-Quenches (8e-5 -> 1e-6) at peak (Ep 11) to freeze weights!</i>"]:::optAStyle
        LossGen --> OptGen["<b>optimizer_gen</b>: AdamW(lr=5e-4, wd=1e-6)<br/><b>CosineAnnealingLR</b> across all 50 Epochs<br/><i>Uninterrupted generation learning</i>"]:::optAStyle
    end

    subgraph OptionC["6. OPTION C: Composite Checkpoint Synthesis"]
        LossCls --> ClsBest["<b>checkpoint_cls_best.pt</b><br/>Peak Cls: Macro-F1 86.03% (Epoch 11)"]:::optCStyle
        LossGen --> GenBest["<b>checkpoint_gen_best.pt</b><br/>Peak Gen: Min NLL / Low PPL (Epoch 48)"]:::optCStyle
        ClsBest & GenBest --> Synth["<b>synthesize_composite_checkpoint()</b><br/>Extracts Head 1 from Ep 11 + Head 2 from Ep 48"]:::optCStyle
        Synth --> CompCkpt["<b>checkpoint_best.pt</b> (Merged Peak State Dict)"]:::optCStyle
    end

    subgraph AutonomousGate["7. End-to-End Autonomous Pipeline"]
        CompCkpt --> Gate{"P(clickbait) ≥ τ*?"}:::gateStyle
        Gate -->|YES: Clickbait| TriggerGen["Trigger Head 2 Beam Search<br/>Produce remediated factual headline"]:::gateStyle
        Gate -->|NO: Genuine| Passthrough["Passthrough original headline<br/>(Identity mapping)"]:::gateStyle
    end
```

---

## 🎯 Model 3 (11): Decoupled 2 × 8 Multi-Head Declickbait System
**Hardware**: Google Colab T4 / Dual T4 | **Scale**: ~28.85M Parameters (~115 MB) | **Strategy**: Option 1 Dual Independent Optimizers with Alternating Steps

```mermaid
flowchart TD
    classDef inputStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef h1Style fill:#0f3832,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef h2Style fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef optStyle fill:#431407,stroke:#fb923c,stroke-width:2px,color:#f8fafc;
    classDef gateStyle fill:#4c0519,stroke:#fb7185,stroke-width:2px,color:#f8fafc;

    subgraph Inputs["1. Unified Trilingual Inputs"]
        Title["Headline Title (max 64 tokens)"]:::inputStyle
        Content["Lead Content (max 512 tokens)"]:::inputStyle
        Target["Reference Truth Title (max 32 tokens)"]:::inputStyle
        BUET["BUET BanglishBERT Tokenizer<br/>(64,768 Vocab)"]:::inputStyle
        Title & Content & Target --> BUET
    end

    subgraph SharedEmb["2. Shared Embedding Layer (~8.29M Params)"]
        BUET --> Emb["Shared Multilingual Embedding (64,768 × 128)<br/><i>Exclusively owned & updated by Head 1 Cls Optimizer</i>"]:::h1Style
        Emb --> PE["Sinusoidal Positional Encoding"]:::h1Style
    end

    subgraph Head1["3. HEAD 1: Evidence Attention Classifier (~9.83M Params)"]
        PE --> TT["Title Tower (3 Transformer Layers, d=128)"]:::h1Style
        PE --> CT["Content Tower (4 Transformer Layers, d=128)"]:::h1Style
        TT & CT --> CrossAttn["Title-Guided Cross-Attention (Q=T, K=C, V=C)<br/>Token-Level Bait Word Detector (min_i cos(T_i, A_ev, i))"]:::h1Style
        CrossAttn --> ClsHead["MLP Fusion Head -> Logits / Probability P(clickbait)"]:::h1Style
    end

    subgraph Head2["4. HEAD 2: BiLSTM Pointer-Generator Rewriter (~19.02M Params)"]
        Emb --> BiLSTM["Content BiLSTM Encoder (hidden=256 -> 512-d)"]:::h2Style
        BiLSTM --> Bahdanau["Bahdanau Additive Coverage Attention"]:::h2Style
        Bahdanau --> PGen["Pointer-Generator Soft Switch: p_gen ∈ [0, 1]"]:::h2Style
        PGen --> Scatter["Dynamic Extended OOV Scatter-Add<br/>P(w) = p_gen·P_vocab + (1-p_gen)·a_i^t"]:::h2Style
        Scatter --> GenOut["Generated Factual Replacement Headline"]:::h2Style
    end

    subgraph AlternatingOpt["5. Alternating Dual-Optimizer Execution"]
        ClsHead --> LossCls["L_cls: Binary Cross-Entropy"]:::optStyle
        GenOut --> LossGen["L_gen: NLL + λ · Coverage Penalty"]:::optStyle
        LossCls --> OptCls["<b>optimizer_cls</b>: AdamW(lr=2e-4, wd=1e-4)<br/>Updates: Embedding + Towers + CrossAttn"]:::optStyle
        LossGen --> OptGen["<b>optimizer_gen</b>: AdamW(lr=5e-4, wd=1e-6)<br/>Updates: BiLSTM + Decoder + Softmax (256x64k)"]:::optStyle
    end

    subgraph AutonomousGate["6. End-to-End Autonomous Pipeline"]
        ClsHead --> Gate{"P(clickbait) ≥ τ*?"}:::gateStyle
        Gate -->|YES: Clickbait| TriggerGen["Trigger Head 2 Beam Search<br/>Produce remediated factual headline"]:::gateStyle
        Gate -->|NO: Genuine| Passthrough["Passthrough original headline<br/>(Identity mapping)"]:::gateStyle
    end
```

---

## 🎯 Model 2 (06_v3): From-Scratch Pointer-Generator Headline Generator
**Hardware**: Google Colab T4 | **Vocab**: BUET BanglishBERT (32k) | **Paradigm**: Hybrid Seq2Seq with Dynamic OOV Copying & Coverage

```mermaid
flowchart TD
    classDef inputStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef encStyle fill:#0f3832,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef decStyle fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef ptrStyle fill:#431407,stroke:#fb923c,stroke-width:2px,color:#f8fafc;
    classDef lossStyle fill:#4c0519,stroke:#fb7185,stroke-width:2px,color:#f8fafc;

    subgraph InputProcessing["1. Text Ingestion & Lead Selection"]
        Raw["Raw News Article Body"]:::inputStyle --> Lead["Lead Sentence Extractor<br/>(Clean Promo/Adverts, first 1500 chars)"]:::inputStyle
        Lead --> BUET_Tok["csebuetnlp/banglishbert Tokenizer<br/>(32,000 Vocab)"]:::inputStyle
        BUET_Tok --> OOV["ExtendedVocabBatchProcessor<br/>(Maps source-specific OOVs to indices 32000+)"]:::inputStyle
    end

    subgraph Encoder["2. Bidirectional LSTM Encoder"]
        OOV --> Emb["Embedding Layer<br/>(128-d)"]:::encStyle
        Emb --> BiLSTM["Bidirectional LSTM (hidden=256)<br/>Pack/Pad sequences"]:::encStyle
        BiLSTM --> EncOut["Encoder Outputs: h_i<br/>(dim: 512)"]:::encStyle
        BiLSTM --> Reduce["State Reducer (tanh(W [h_fwd; h_bwd]))<br/>Initializes Decoder dec_h0, dec_c0"]:::encStyle
    end

    subgraph Decoder["3. Unidirectional LSTM Decoder with Coverage"]
        Reduce --> DecLSTM["Unidirectional LSTM<br/>(hidden=256)"]:::decStyle
        DecLSTM --> DecState["Decoder State s_t<br/>(dim: 256)"]:::decStyle
        EncOut & DecState & CovPrev["Coverage History<br/>c^t = sum(a^t')"]:::decStyle --> Bahdanau["Bahdanau Additive Attention<br/>v^T tanh(W_h h_i + W_s s_t + W_c c^t)"]:::decStyle
        Bahdanau --> AttnDist["Attention Distribution a^t"]:::decStyle
        Bahdanau --> Context["Context Vector c_t*<br/>sum(a_i^t h_i)"]:::decStyle
    end

    subgraph PointerSwitch["4. Dynamic Pointer-Generator Copy Mechanism"]
        Context & DecState & InEmb["y_emb (prev token)"]:::ptrStyle --> PGenLinear["Linear + Sigmoid<br/><b>p_gen ∈ [0, 1]</b>"]:::ptrStyle
        Context & DecState --> VocabDist["Linear + Softmax<br/>P_vocab (32,000)"]:::ptrStyle
        
        PGenLinear & VocabDist --> GenMass["Generation Mass<br/>p_gen × P_vocab(w)"]:::ptrStyle
        PGenLinear & AttnDist --> CopyMass["Copy Mass<br/>(1 - p_gen) × a_i^t"]:::ptrStyle
        
        GenMass & CopyMass --> Scatter["Scatter-Add into Extended Vocab<br/>P(w) = p_gen·P_vocab + (1 - p_gen)·sum(a_i^t)"]:::ptrStyle
        Scatter --> FinalDist["Final Probability Distribution<br/>(32k + dynamic batch OOVs)"]:::ptrStyle
    end

    subgraph LossOpt["5. Loss Function & Beam Search"]
        FinalDist --> NLL["NLL Loss<br/>-log P(y_t*)"]:::lossStyle
        AttnDist & CovPrev --> CovPenalty["Coverage Loss Penalty<br/>min(a_i^t, c_i^t)"]:::lossStyle
        NLL & CovPenalty --> TotalLoss["Total Loss<br/><b>L = L_NLL + λ · L_cov</b>"]:::lossStyle
        FinalDist --> Beam["Beam Search Decoding<br/>Trigram Blocking + Coverage Penalty β"]:::lossStyle
    end
```

---

## 🎯 Model 1 (02_v3): Title-Guided Evidence Attention Two-Tower Classifier
**Hardware**: Dual T4 / CPU Edge | **Vocab**: Byte-Level BPE (16k) | **Footprint**: ~3.59M Params (<15ms)

```mermaid
flowchart TD
    classDef inputStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef tokStyle fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef embStyle fill:#311042,stroke:#c084fc,stroke-width:2px,color:#f8fafc;
    classDef towerStyle fill:#0f3832,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef attnStyle fill:#431407,stroke:#fb923c,stroke-width:2px,color:#f8fafc;
    classDef featStyle fill:#172554,stroke:#60a5fa,stroke-width:2px,color:#f8fafc;
    classDef headStyle fill:#4c0519,stroke:#fb7185,stroke-width:2px,color:#f8fafc;

    subgraph Inputs["1. Trilingual Inputs"]
        T["Headline / Title<br/>(max 64 tokens)"]:::inputStyle
        C["Article Body / Content<br/>(max 768 tokens)"]:::inputStyle
    end

    subgraph Tokenizer["2. Byte-Level BPE (16,000 Vocab, 0% OOV)"]
        T --> BPE_T["Byte-Level BPE Tokenizer"]:::tokStyle
        C --> BPE_C["Byte-Level BPE Tokenizer"]:::tokStyle
        BPE_T --> T_IDs["t_ids, t_mask"]:::tokStyle
        BPE_C --> C_IDs["c_ids, c_mask"]:::tokStyle
    end

    subgraph Embedding["3. Shared Embedding & Positional Encoding"]
        T_IDs --> EMB["Shared Embedding Table<br/>(16,000 × 128)"]:::embStyle
        C_IDs --> EMB
        EMB --> PE["Sinusoidal Positional Encoding"]:::embStyle
    end

    subgraph Towers["4. Decoupled Two-Tower Encoders"]
        PE --> TT["Title Tower L1<br/>(3 Transformer Layers, d=128)"]:::towerStyle
        PE --> CT["Content Tower L1<br/>(4 Transformer Layers, d=128)"]:::towerStyle
        TT --> T_repr["T_emb (Headline Repr)"]:::towerStyle
        CT --> C_repr["C_emb (Content Repr)"]:::towerStyle
    end

    subgraph EvidenceCrossAttn["5. Title-Guided Evidence Cross-Attention"]
        T_repr -->|Query| MHA["Cross-Attention MultiHeadAttn<br/>(Q = T, K = C, V = C)"]:::attnStyle
        C_repr -->|Key, Value| MHA
        MHA --> A_ev["A_ev (Headline-Aligned Evidence Repr)"]:::attnStyle
    end

    subgraph FeatureFusion["6. Semantic Pooling & Discrepancy Features"]
        T_repr --> P1["Mean Pool: h_title<br/>(128)"]:::featStyle
        A_ev --> P2["Mean Pool: h_evidence<br/>(128)"]:::featStyle
        C_repr --> P3["Mean Pool: h_content_global<br/>(128)"]:::featStyle
        P1 & P2 --> Diff["|h_title - h_evidence|<br/>(128)"]:::featStyle
        P1 & P2 --> Prod["h_title ⊙ h_evidence<br/>(128)"]:::featStyle
        P1 & P2 --> CosE["cos(h_title, h_evidence)<br/>(1)"]:::featStyle
        P1 & P3 --> CosG["cos(h_title, h_content_global)<br/>(1)"]:::featStyle
        T_repr & A_ev --> Bait["min_i cos(T_i, A_ev, i)<br/>Token-Level Bait Detector (1)"]:::featStyle
    end

    subgraph Head["7. Fusion MLP & Classification Head"]
        P1 & P2 & P3 & Diff & Prod & CosE & CosG & Bait --> Cat["Concat Features<br/>(Dim: 128*5 + 3 = 643)"]:::headStyle
        Cat --> Linear1["Linear(643 -> 128) + GELU + Dropout"]:::headStyle
        Linear1 --> Clf["Linear(128 -> 64) -> GELU -> Linear(64 -> 1)"]:::headStyle
        Clf --> Logits["Logits / Probability<br/><b>Clickbait vs Factual</b>"]:::headStyle
    end
```

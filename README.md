# CinLM — PT-BR Hybrid Small Language Model (Interpretable by Design)

**CinLM** é um modelo de linguagem pequeno (~100M parâmetros ativos / ~260M totais) focado em **Português do Brasil (PT-BR)**, treinado em dados 100% livres e projetado desde o dia 0 com dois objetivos combinados:

1. **Alta competência por parâmetro (homemade em GPU consumer / Colab G4):**
   - **Hybrid Attention 3:1** — 12 camadas de *Sliding Window Attention* (janela 256) + 4 camadas de *Atenção Causal Global* (GQA 5:1, RoPE, QK-RMSNorm).
   - **Engram Conditional Memory** (inspirado em `arXiv:2601.07372`) nas camadas 1 e 5 — lookup $O(1)$ de n-gramas (bigramas + trigramas, multi-head hashing) com gate escalar contextual $\alpha_t \in (0, 1)$ e convolução causal depthwise.
   - **Fine-Grained Sparse `ReLU²` MoE** — 1 *Shared Expert* + 8 *Routed Experts* (top-2), execução esparsa real por token e balanceamento sem *auxiliary loss* (bias dinâmico estilo DeepSeek-V3/V4).
2. **Interpretabilidade nativa por Frontier LLMs:**
   - **Residual stream único e aditivo** (`emb + engram + attn + moe`) → **Direct Logit Attribution (DLA)** exato (soma das contribuições iguala exatamente o logit do token previsto) + **Logit Lens** camada a camada.
   - **Ativações `ReLU²`** naturalmente esparsas (~80–90% zeros) para reduzir polissemança.
   - **Gates explícitos do Engram**, **mapas de atenção separados (SWA vs. Global)** e **estatísticas de roteamento de experts** exportados via `run_frontier_audit()` em JSON pronto para um Frontier LLM auditar e propor intervenções causais (`InterventionSpec`).

---

## Escada de Ablação Iso-Ativa (~100M ativos/token)

Todas as variantes de 100M usam o mesmo tokenizador BPE PT-BR (`vocab=32.768`), `d_model=640`, `n_layers=16`, `n_heads=10`, `n_kv_heads=2`, `context=2048` e *weight tying*:

| Preset | Atenção | Engram | FFN / MoE | Total Params (exato) | Ativos / Token | O que isola |
|---|---|---|---|---:|---:|---|
| `dense_global_100m` | 16× Global | Não | Dense `ReLU²` (3072) | 99.64M | 99.64M | Baseline Transformer puro |
| `dense_hybrid_100m` | 12× SWA + 4× Global | Não | Dense `ReLU²` (3072) | 99.64M | 99.64M | Ganho/custo de Hybrid Attention |
| `hybrid_engram_100m` | 12× SWA + 4× Global | Sim (L1, L5) | Dense `ReLU²` (3072) | 134.54M | 100.96M | Ganho de memória estática Engram |
| `full_260m_a100m` | 12× SWA + 4× Global | Sim (L1, L5) | MoE `ReLU²` (1s + 8r top-2) | 260.45M | 101.04M | Arquitetura completa CinLM |

---

## Receita de Dados 100% Livres (PT-BR)

1. **Pré-treino Base (70%):** `Polygl0t/gigaverbo-v2` (streaming direto do Hugging Face) + opcional `HuggingFaceFW/fineweb-edu` (`sample-10BT`, 15%).
2. **Mid-training Educacional / Sintético (15–30%):** `Polygl0t/gigaverbo-v2-synth`.
3. **SFT Conversacional PT-BR:** `Polygl0t/gigaverbo-v2-sft`, `dominguesm/Canarim-Instruct-PTBR-Dataset`, `FreedomIntelligence/evol-instruct-portuguese`, `CohereForAI/aya_dataset` (Portuguese).

---

## Como Rodar

### 1. Instalar e rodar testes de integridade
```bash
pip install -e ".[dev]"
pytest -v
```

### 2. Treinar o tokenizador BPE PT-BR (32k)
```bash
python scripts/train_tokenizer.py --max-docs 200000 --vocab-size 32768 --output artifacts/cinlm_bpe_32k.json
```

### 3. Smoke test rápido ou treino na G4 do Colab
```bash
# Smoke test sintético (verifica throughput e gradientes)
python scripts/train.py --preset full_260m_a100m --synthetic --seq-len 1024 --batch-size 4 --grad-accum 1 --max-steps 20

# Pré-treino real com streaming do GigaVerbo-v2 na G4 (96 GB VRAM)
python scripts/train.py \
  --preset full_260m_a100m \
  --tokenizer artifacts/cinlm_bpe_32k.json \
  --seq-len 2048 --batch-size 16 --grad-accum 4 \
  --max-steps 20000 --compile --out-dir runs/cinlm_full_ptbr
```

### 4. Auditoria de interpretabilidade para Frontier LLM
```bash
python scripts/audit_prompt.py \
  --prompt "A capital do Brasil é" \
  --tokenizer artifacts/cinlm_bpe_32k.json \
  --checkpoint runs/cinlm_full_ptbr/ckpt_step_20000.pt \
  --out audit.json
```
Notebook pronto para Google Colab G4: `notebooks/CinLM_Colab_G4.ipynb`.

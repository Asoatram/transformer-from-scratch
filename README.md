# Implementasi Transformer dari Nol

**Muhamad Daffa Azfa Rabbani (22/503970/TK/55101)**

Implementasi arsitektur GPT-style Transformer menggunakan NumPy

## Deskripsi

Proyek ini mengimplementasikan komponen inti Transformer untuk autoregressive language modeling, termasuk:

- Token embedding dengan scaling
- Sinusoidal positional encoding
- Scaled dot-product attention
- Multi-head attention (8 heads)
- Feed-forward network dengan aktivasi GELU
- Layer normalization (pre-norm architecture)
- Residual connections
- Causal masking untuk autoregressive generation

## Dependensi

```bash
numpy
io
re
collections
```

## Instalasi

```bash
pip install numpy io re collections
```

## Penggunaan

Jalankan notebook Jupyter:

```bash
jupyter notebook transformer.ipynb
```

Atau konversi ke script Python:


## Struktur Kode

- **Section 1**: Text Preprocessing
- **Section 2**: Tokenizer
- **Section 3**: Word Embedding
- **Section 4**: Self Attention
- **Section 5**: Causal Mask
- **Section 6**: Multi Head Attention
- **Section 7**: Feed Forward Network
- **Section 8**: Layer Norm
- **Section 9**: Decoder Block
- **Section 10**: Transformer



## Testing

Program mencakup pengujian untuk:
- Validasi dimensi tensor
- Verifikasi properti softmax
- Validasi causal masking

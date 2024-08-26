# Transformer Implementation in PyTorch: "Attention is All You Need"

This repository contains a PyTorch implementation of the Transformer model as described in the seminal paper "Attention is All You Need". The Transformer model, introduced by Vaswani et al. in 2017. I have tried implementing it with (only) PyTorch by breaking down different parts of the paper. The transformers paper was the base of almost all the initial LLMs models such as GPT(Generative Pretrained Transformers).
## Table of Contents

- [Overview](#overview)
- [Components](#components)
  - [Input Embedding](#input-embedding)
  - [Positional Encoding](#positional-encoding)
  - [Multi-Head Attention](#multi-head-attention)
  - [Feed-Forward Network](#feed-forward-network)
  - [Residual Connection and Layer Normalization](#residual-connection-and-layer-normalization)
  - [Encoder and Decoder](#encoder-and-decoder)
  - [Final Projection Layer](#final-projection-layer)
- [Building the Transformer](#building-the-transformer)
- [Usage](#usage)
- [References](#references)

## Overview

The Transformer model consists of an encoder-decoder structure. The encoder processes the input sequence to create a representation, which the decoder uses to generate the output sequence. Both the encoder and decoder use multi-head self-attention and feed-forward neural networks, connected with residual connections and layer normalization.

## Components

### 1. Input Embedding

The `Input_Embedding` class converts words into dense vectors of a fixed dimension (`d_model`). The embedding is scaled by the square root of `d_model` to adjust the variance:

```python
class Input_Embedding(nn.Module): 
    def __init__(self, vocab_size:int, d_model:int ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

### 2. Positional Encoding

Positional encoding injects information about the position of words in a sentence. It uses sine and cosine functions to generate different frequencies for each dimension:

```python

class Positional_Encoding(nn.Module):
    def __init__(self, d_model:int, sequence_length: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos_enc = torch.zeros(sequence_length, d_model)
        positions = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(positions * denominator)
        pos_enc[:, 1::2] = torch.cos(positions * denominator)
        self.register_buffer("postitons", pos_enc.unsqueeze(0))

    def forward(self, x):
        x += self.pos_enc[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
```

### 3. Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces. The attention mechanism computes a weighted sum of values (V), where the weight assigned to each value is determined by a similarity score between the query (Q) and key (K):

```python

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h: int, dropout: float):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -10e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return attention_scores @ value, attention_scores
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)
```

### 4. Feed-Forward Network

Each encoder and decoder layer contains a feed-forward neural network that is applied to each position separately and identically:

```python

class Feed_Forward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_layer_1 = nn.Linear(d_model, d_ff)
        self.linear_layer_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear_layer_2(self.dropout(torch.relu(self.linear_layer_1(x))))
```

### 5. Residual Connection and Layer Normalization

Residual connections wrap around the sub-layers of each encoder and decoder layer, followed by layer normalization:

```python

class Residual_Connection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = Layer_Normalization(features)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

### 6. Encoder and Decoder

The encoder and decoder are composed of a stack of identical layers:

```python

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = Layer_Normalization(features)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

### 7. Final Projection Layer

The final output is passed through a linear transformation layer to project the hidden states to the vocabulary size:

```python

class Projection_layer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return self.projection(x)
```

## Building the Transformer

The build_transformer function initializes and returns a Transformer model:

```python

def build_transformer(source_vocab_size, target_vocab_size, source_sequence_length, target_sequence_length, d_model=512, N_layers=6, h=8, dropout=0.1, d_ff=2048):
    source_embeddings = Input_Embedding(vocab_size=source_vocab_size, d_model=d_model)
    target_embeddings = Input_Embedding(vocab_size=target_vocab_size, d_model=d_model)
    source_pos = Positional_Encoding(d_model, source_sequence_length, dropout)
    target_pos = Positional_Encoding(d_model, target_sequence_length, dropout)

    encoder_blocks = [Encoder_Block(d_model, MultiHeadAttention(d_model, h, dropout), Feed_Forward(d_model, d_ff, dropout), dropout) for _ in range(N_layers)]
    decoder_blocks = [Decoder_Block(d_model, MultiHeadAttention(d_model, h, dropout), MultiHeadAttention(d_model, h, dropout), Feed_Forward(d_model, d_ff, dropout), dropout) for _ in range(N_layers)]
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    projection_layer = Projection_layer(d_model, target_vocab_size)
    
    transformer = Transformer(encoder, decoder, source_embeddings, target_embeddings, source_pos, target_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
```
## Usage
  - Create a virtual environment
  - If you have [uv](https://astral.sh/blog/uv-unified-python-packaging) installed:
    -  ```uv venv```
    -  Activate with: ```.venv\scripts\activate```
  - Create without uv:
    -  ```python -m venv venv```
    
  
- Install PyTorch: PyTorch Installation
- Clone the repository: ```git clone https://github.com/yourusername/transformer-pytorch.git```
- Import the `build_transformer` function and initialize the model with your desired configuration.
- Train the model using your dataset.
  - Example dataset: [IIT Bombay English-Hindi Translation Dataset](https://www.kaggle.com/datasets/vaibhavkumar11/hindi-english-parallel-corpus)

## References
- [Umar Jamil's Implementation](https://youtu.be/ISNdQcPhsts?si=hqGDqjfqUiCdF5xx)
- [Attention Is All you Need paper](https://arxiv.org/abs/1706.03762)

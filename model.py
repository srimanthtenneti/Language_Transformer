import torch
import torch.nn as nn
import torch.nn.functional as f

torch.manual_seed(1080)

batch_size = 64
block_size = 256
n_embed = 384
head_size = 16
num_epochs = 10000
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4

### Fetching Batches 

def get_batch(split) :
  ### Getting the train or valid data
  data = train_data if split == 'train' else valid_data
  ix = torch.randint((len(data) - block_size), (batch_size, ))
  x = torch.stack([data[i : i + block_size] for i in ix])
  y = torch.stack([data[i+1 : i+block_size + 1] for i in ix])
  return x, y

### Self Attention Head

class self_attention_head (nn.Module) :
  def __init__(self, head_size) :
    super().__init__()
    self.key   = nn.Linear(n_embed, head_size, bias= False)
    self.query = nn.Linear(n_embed, head_size, bias= False)
    self.value = nn.Linear(n_embed, head_size, bias= False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x) :

    B, T, C = x.shape

    k = self.key(x) # (B, n_embed, head_size) (B, n_embed, C)
    q = self.query(x) # (32, head_size)
    v = self.value(x) # (32, head_size)

    wei = q @ k.transpose(-2, -1) * head_size**-0.5 # B, T, head_size

    ### Masking the upper triangle
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = f.softmax(wei, dim = -1)
    out = wei @ v
    return out

### Multi-Head Attention 

class Multi_Head_Attention(nn.Module) :

  def __init__(self, n_heads, head_size) :
    super().__init__()
    self.heads = nn.ModuleList([self_attention_head(head_size) for _ in range(n_heads)])
    self.proj = nn.Linear(n_embed, n_embed)

  def forward(self, x) :
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.proj(out)
    return out


class FF (nn.Module) :
  def __init__(self, n_embed) :
    super().__init__()

    self.net = nn.Sequential(
        nn.Linear(n_embed, 4 * n_embed),
        nn.ReLU(),
        nn.Linear(4 * n_embed, n_embed),
        nn.Dropout(0.4)
        )
  def forward(self, x) :
    return self.net(x)
  
  class Block(nn.Module) :
    def __init__(self, n_embed, n_head) :
      super().__init__()

      head_size = int(n_embed / n_head)
      self.sa = Multi_Head_Attention(n_head, head_size)
      self.ffwd = FF(n_embed)

      self.ln1 = nn.LayerNorm(n_embed)
      self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x) : ### Added Residual Connections for better gradient flow
      x = x + self.sa(self.ln1(x))
      x = x + self.ffwd(self.ln2(x))
      return x
    
    import torch.nn as nn
import torch.nn.functional as f

class LM (nn.Module) :

  def __init__(self) :
    super().__init__()
    ### Token Embedding table of (vocab_size, vocab_size)
    ### Defining a 32 dimensional embedding
    self.token_embedding = nn.Embedding(vocab_size, n_embed)
    self.position_embedding = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(
         Block(n_embed, n_head = 4),
         Block(n_embed, n_head = 4),
         Block(n_embed, n_head = 4),
         Block(n_embed, n_head = 4),
         Block(n_embed, n_head = 4),
         Block(n_embed, n_head = 4),
         nn.LayerNorm(n_embed),
    )
    self.ln_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets = None) :
    ### Logits -> Scores for next characters in the sequence
    ### This embedding table is going to give us a token embedding
    B, T = idx.shape
    token_emb = self.token_embedding(idx) # Batch, Time, Channel (n_embed)
    ### Now we define another embedding layer - > Positional embedding
    pos_emb = self.position_embedding(torch.arange(T, device=device)) # T, C
    x = token_emb + pos_emb # (B, T, C) + (T, C) -> token emb will be broadcasted and added
    x = self.blocks(x)
    logits    = self.ln_head(x) # Batch, Time, Channel (vocab_size )

    if targets == None :
      loss = None

    else :
      B, T, C = logits.shape
      logits = logits.view(B * T, C) ### Joining the batch and time dimensions
      targets = targets.view(B * T)

      loss = f.cross_entropy(logits, targets) # -> Expects (B, C)
    return logits, loss

  def generate(self, idx, max_new_tokens) :
    ### idx is of dim (B, T) array of indices in current context
    for _ in range(max_new_tokens) :
      idx_cond = idx[:, -block_size:] ### Cropping the last block size of tokens
      logits, loss = self(idx_cond)
      ### Focus only on last time step
      logits = logits[ : , -1, :] # Becomes B, C
      probs = f.softmax(logits,  dim = -1)
      ### Sample from Distribution
      inx_next = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx, inx_next), dim = 1) # (B, T + 1)
    return idx

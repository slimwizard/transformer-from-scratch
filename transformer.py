import torch
import torch.nn as nn
import math

############################################
############### INPUT ######################
############################################


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model  # dim of input embedding vector
        self.vocab_size = vocab_size  # number of words in vocabulary

        # pytorch already has an Embedding class to leverage
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PostitionalEncoding(nn.Module):
    """
    the positional encodings are only defined/computed once and
    reused for each input embedding for both training and inference
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len  # sequence length represents the maximum length of an input sentence
        self.dropout = nn.Dropout(dropout)

        """
        We need positional vectors of size d_model, because each positional embedding needs to match the input vector size. 
        We need to account for sequence length because we need a positinal embedding vector for all possible input length,
        meaning we need to account for everything up to the maximum size of input.

        So we will create a matrix of shape (seq_len, d_model)
        """
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1) which will represent the position of the word in the sentence
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        """this vector represents the term that the position is divided by in the PE formula
           Note that this implementation differs slightly from paper as the denomitator is calculated in log space"""
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even positions, cos to odd positions

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add new dimension to the positional encoding matrix (pe) to account for batching
        pe = pe.unsqueeze(0)  # will result in a matrix of shape (1, seq_len, d_model)

        # register this tensor in the "buffer" of the model
        # this will allow the value to be saved along with the state of the model when exporting
        self.register_buffer("pe", pe)

    def forward(self, x):
        # add positinal encoding to every word in the sentence
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


############################################
############## ENCODER #####################
############################################

"""
Add & Norm = Layer Normalization
Meaning:
if you have a batch of n items, for each item in this batch we calculate mean and variance independent of other items in the batch, 
then we calculate new values for each item in the batch based on their own mean and variance. Gamma (or alpha) and beta (or bias) are also typically introduced
in layer normalization. One is multiplied to each feature in an item and the other is added to each feature in an item. This helps the model 
'amplify' the values it wants to be amplified
"""


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        # epsilon is needed in the normalization function in case sigma is very close to, or is, zero i.e. numerical stability
        self.eps = eps
        # using nn.Parameter() makes the parameter learnable
        self.alpha = nn.Parameter(torch.ones(1))  # alpha is multiplied
        self.bias = nn.Parameter(torch.ones(1))  # bias is added

    def forward(self, x):
        # calculate across last dimension, as the first dimension will be the batch
        mean = x.mean(dim=-1, keep_dim=True)
        std = x.std(dim=-1, keep_dim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


"""
Feed Forward
Fully Connected Layer that the model uses both in the encoder and decoder
This layer is 2 matrices (w1 and w2) which are multiplied by x and a bias is added (b1 and b2), one after the other, with a ReLU in between
We can do this in pytorch with a Linear layer

d_ff represents one shape dimension of each matrix (i.e. matrix 1 shape (d_ff, d_model) and matrix 2 shape (d_model, d_ff)) 
and the value which is used in the Transformer paper is 2048
"""


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # with nn.Linear layers, the "bias" argument (i.e. nn.Linear(1, 2, bias=True)) defaults to True, so we don't need to specify that
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

        self.dropout = dropout

    def forward(self, x):
        # shape transformation from input to output will look like:
        # (batch, seq_len, d_model) --Linear1--> (batch, seq_len, d_ff) --Linear2--> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


"""
Multi-Head Attention (MHA)

Takes input from encoder and uses it 3 times: query, key, and values

input (seq_len, d_model) -> Query (seq_len, d_model)
                         -> Key   (seq_len, d_model)
                         -> Value (seq_len, d_model)

There are 3 weight matrices of size (d_model, d_model), one for each of the attention types: Wq, Wk, Wv
Q, K, and V are multiplied by Wq, Wk, and Wv respectively, to produce Q', K', and V'

Q', K', and V' are split along the embedding dimension based on the number of attetion heads (h)
So if h=5 then each attention type embedding will be split into 5 uniform submatrices of size (seq_len, dk) - add them all back together and they are size (seq_len, d_model)
So dk * h should equal d_model

This means that each head has access to the full sentence but sees a different slice of the embedding from what the other heads see

Now, apply the attention formula to each set of head matrices, each of which results in a final head matrix of size (seq_len, dv) 
!! NOTE THAT dv and dk are exactly the same, the switch from a k to a v after running the attention formula is just a result of the algebraic letters being used
!! so h*dv = h*dk = d_model

Finally, concat the heads back into a single matrix of size (seq_len, h*dv) and multiply by another weight matrix Wo, size (h*dv, d_model), 
to get the final MHA output which is the same dimensions as the input (seq_len, d_model)
"""


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self, seq_len: int, d_model: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()

        # d_model must be divisible by num_heads in order to get an even split across the original matrix
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # build weight matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        """
        Mask if used when we don't want some words to interact with other words
        """

        # going from size (batch, seq_len, d_model) to (batch, seq_len, d_model)
        query = self.w_q(q)  # represents Q'
        key = self.w_k(k)  # represents K'
        value = self.w_v(v)  # represents V'

        q_heads = {}
        k_heads = {}
        v_heads = {}
        for head in range(1, self.num_heads + 1):
            q_heads[f"head{head}"] = query[:, :, self.d_k * head - 1 : self.d_k * head]
            k_heads[f"head{head}"] = key[:, :, self.d_k * head - 1 : self.d_k * head]
            v_heads[f"head{head}"] = value[:, :, self.d_k * head - 1 : self.d_k * head]

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k)

        print("QUERY HEADS from loop")
        print(q_heads)

        print("QUERY HEADS from view")
        print(query)

        # print(k_heads)
        # print(key)

        # print(v_heads)
        # print(value)

        # return ((q_heads, query), (k_heads, key), (v_heads, value))
        return None


############################################
############## DECODER #####################
############################################

"""


"""

############################################
############### OUTPUT #####################
############################################


############################################
############# TRANSFORMER ##################
############################################


class Model(nn.Module):
    def __init__(
        self, d_model: int, seq_len: int, vocab_size: int, dropout: float
    ) -> None:
        super().__init__()

        self.ie = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
        self.pe = PostitionalEncoding(d_model=d_model, seq_len=seq_len, dropout=dropout)

    def forward(self, x):
        # Forward pass through your custom modules
        x = self.ie(x)
        x = self.pe(x)

        return x


class MHAOnlyModel(nn.Module):
    def __init__(
        self, seq_len: int, d_model: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()

        self.mha = MultiHeadAttentionBlock(
            seq_len=seq_len, d_model=d_model, num_heads=num_heads, dropout=dropout
        )

    def forward(self, x):
        x = self.mha(x, x, x, 0)

        return x


############################################

# model = Model(d_model=8, seq_len=24, vocab_size=1024, dropout=0.0)

# input_tensor = torch.randint(low=0, high=100, size=(2, 8))
# print("input tensor: ", input_tensor)


# print("output tensor: ", model(input_tensor))

mha_model = MHAOnlyModel(seq_len=12, d_model=4, num_heads=2, dropout=0.0)

input_tensor = torch.rand(size=(1, 12, 4))

print("input tensor: ", input_tensor, "shape: ", input_tensor.shape)


print("output tensor: ", mha_model(input_tensor))

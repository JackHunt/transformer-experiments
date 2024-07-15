import torch.nn as nn

from attention_functions import scaled_dot_product


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        self.Q = nn.Linear(self.d_model, self.d_model)
        self.K = nn.Linear(self.d_model, self.d_model)
        self.V = nn.Linear(self.d_model, self.d_model)
        self.fc = nn.Linear(self.d_model, self.d_model)

        self.attention_fn = scaled_dot_product

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output, attn = self.attention_fn(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(output)
        return output, attn

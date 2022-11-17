from torch import nn
import torch

import sys
sys.path.append('..')
sys.path.append('../..')

import math



# Example parameters: shape=(28, 28), bands=8
# encoding_size = bands*2*2
def fourier_features(shape, bands):
    # This first "shape" refers to the shape of the input data, not the output of this function
    dims = len(shape)
    # Every tensor we make has shape: (bands, dimension, x, y, etc...)

    # Pos is computed for the second tensor dimension
    # (aptly named "dimension"), with respect to all
    # following tensor-dimensions ("x", "y", "z", etc.)
    pos = torch.stack(list(torch.meshgrid(
        *(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape))
    )))
    pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

    # Band frequencies are computed for the first
    # tensor-dimension (aptly named "bands") with
    # respect to the index in that dimension
    band_frequencies = (torch.logspace(
        math.log(1.0),
        math.log(shape[0]/2),
        steps=bands,
        base=math.e
    )).view((bands,) + tuple(1 for _ in pos.shape[1:])).expand(pos.shape)

    # For every single value in the tensor, let's compute:
    #             freq[band] * pi * pos[d]

    # We can easily do that because every tensor is the
    # same shape, and repeated in the dimensions where
    # it's not relevant (e.g. "bands" dimension for the "pos" tensor)
    result = (band_frequencies * math.pi * pos).view((dims * bands,) + shape)

    # Use both sin & cos for each band, and then add raw position as well
    result = torch.cat([
        torch.sin(result),
        torch.cos(result),
    ], dim=0)

    # return nn.Parameter(result)
    return result



# =============================================================================
# Token processing blocks
# =============================================================================

# =============================================================================
# Processing blocks
# X-attention input
#   Q/z_input         -> (#tokens_Q, batch_size, embed_dim)
#   K/V/x             -> (#tokens_V, batch_size, embed_dim)
#   (mask) key_padding_mask  -> (batch_size, #tokens)
#   (q_mask) attn_mask       -> (#Q, #tokens) / (B*num_heads, #Q, #tokens)
# output -> (#latent_embs, batch_size, embed_dim)
# =============================================================================    
class AttentionBlock(nn.Module):    # PerceiverAttentionBlock
    def __init__(self, opt_dim, heads, dropout, att_dropout, 
                 cross_att=False, 
                 # LSA=False, 
                 add_bias_kv=True, 
                  # **args
                 ):
        super(AttentionBlock, self).__init__()

        self.cross_att = cross_att        
        self.heads = heads        
        # self.LSA = LSA        

        self.layer_norm_x = nn.LayerNorm([opt_dim])
        if cross_att: self.layer_norm_z = nn.LayerNorm([opt_dim])
        self.layer_norm_att = nn.LayerNorm([opt_dim])
        
        self.attention = nn.MultiheadAttention(
        # self.attention = MultiheadAttention(
            opt_dim,            # embed_dim
            heads,              # num_heads
            dropout=att_dropout,
            bias=True,
            add_bias_kv=add_bias_kv,
        )
            
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(opt_dim, opt_dim)
        self.layer_norm_2 = nn.LayerNorm([opt_dim])     # Added in the last version
        self.linear2 = nn.Linear(opt_dim, opt_dim)
        # self.linear3 = nn.Linear(opt_dim, opt_dim)
        


    #   Q/z_input         -> (#tokens_Q, batch_size, embed_dim)
    #   K/V/x             -> (#tokens_V, batch_size, embed_dim)
    #   (mask) key_padding_mask  -> (batch_size, #tokens)
    #   (q_mask) attn_mask       -> (#Q, #tokens) / (B*num_heads, #Q, #tokens)
    def forward(self, x, z, mask=None, q_mask=None):    # , **args
        
        # Norm
        shortcut = x if not self.cross_att else z
        x = self.layer_norm_x(x)
        if self.cross_att: z = self.layer_norm_z(z)
        else: z = x
        
        # Attention
        z_att, _ = self.attention(z, x, x, key_padding_mask=mask, attn_mask=q_mask)  # Q, K, V

        # Add + Norm 
        z_att = shortcut + z_att
        z_att = self.layer_norm_att(z_att)

        # MLP
        z_ff = self.dropout(z_att)
        z_ff = self.linear1(z_ff)
        z_ff = torch.nn.GELU()(z_ff)

        z_ff = self.layer_norm_2(z_ff)
        z_ff = self.dropout(z_att)
        z_ff = self.linear2(z_ff)
        
        # Add
        return z_att + z_ff


# Block with an initial attention layer and an optional set of attention layers.
# Option to return the tokens processed at each layer (return_intermediate_results)
# Option to receive previous token information from a skip connection (get_intermediate_results)
class TransformerBlock(nn.Module):
    def __init__(self, opt_dim, latent_blocks, dropout, att_dropout, heads, cross_heads, 
                 cross_att,
                 add_bias_kv=True, 
                 # LSA=False, 
                 return_intermediate_results = False,
                 get_intermediate_results = False, opt_dim_new = None, 
                 **args):
        super(TransformerBlock, self).__init__()

        self.cross_att = cross_att
        self.latent_blocks = latent_blocks
        self.opt_dim = opt_dim
        self.return_intermediate_results = return_intermediate_results
        self.get_intermediate_results = get_intermediate_results
        
        self.initial_attention = AttentionBlock(opt_dim=opt_dim, heads=cross_heads, dropout=dropout, 
                            att_dropout=att_dropout, cross_att=cross_att,
                            add_bias_kv = add_bias_kv, # LSA=LSA,
                            )
        self.latent_attentions = nn.ModuleList([
            AttentionBlock(opt_dim=opt_dim, heads=heads, dropout=dropout, att_dropout=att_dropout, cross_att=False, 
                           add_bias_kv = add_bias_kv, # LSA=LSA,
                           ) for i in range(latent_blocks)
        ])
        if get_intermediate_results:
            self.process_prev_tokens_layers = nn.ModuleList([
                get_block('MLP', {**{"init_layers": [f"ff_{opt_dim}_gel"], 'ipt_dim': opt_dim_new, 'embed_dim': opt_dim}}) for _ in range(latent_blocks)
                ])
        
    # Add information from an skip connection
    def add_previous_tokens(self, z, prev_inter_tokens, num_block):
        z = torch.cat([z, prev_inter_tokens[num_block]], axis=-1)
        z = self.process_prev_tokens_layers[num_block](z)
        return z
        
    def forward(self, x_input, 
                z = None,
                mask=None,          # key_padding_mask -> ignore 'key' elements (batch_size x number of tokens)
                mask_time_t=None,   # mask to roll back to the original latent vectors for empty time-steps (B,)
                prev_inter_tokens = None,
                ):
            
        # If CrossAttention -> set Q (z) as latent vectors
        if not self.cross_att:  z = x_input
        
        # Initialize results as zero and remove empty samples from QKV
        res = torch.zeros_like(z)
        if self.return_intermediate_results: inter_res = { k: torch.zeros_like(z) for k in range(self.latent_blocks+1) }
        else: inter_res = None
    
        if mask_time_t is not None: 
            x_input = x_input[:, ~mask_time_t]
            z = z[:, ~mask_time_t]
            mask = mask[~mask_time_t]
            # To store intermediate results
        
        # Process data
        # Process with initial Cross/Self-Attention
        z = self.initial_attention(x_input, z, mask=mask)
        if self.return_intermediate_results: inter_res[0][:, ~mask_time_t] = z
        
        # Process with subsequent Self-Attention
        for num_block, latent_attention in enumerate(self.latent_attentions):
            if prev_inter_tokens is not None: z = self.add_previous_tokens(z, prev_inter_tokens, num_block)
            z = latent_attention(z, z, mask=None if self.cross_att else mask)
            if self.return_intermediate_results: inter_res[num_block+1][:, ~mask_time_t] = z
            
        if mask_time_t is not None: 
            # Update results of non-empty samples
            res[:, ~mask_time_t] = z
        else: 
            res = z
        return res, inter_res



# Feed Forward Net
class MLPBlock(nn.Module):
    def __init__(self, ipt_dim, embed_dim, init_layers, 
                 add_x_input=False, dropout=0.0, **args):   #, num_layers
        super(MLPBlock, self).__init__()
        self.embed_dim = embed_dim
        self.add_x_input = add_x_input
        self.dropout = dropout
        if self.dropout > 0.0: self.dropout = nn.Dropout(p=dropout)
        self.seq_init = self._get_sequential_block(init_layers, ipt_dim)
            
    def _get_sequential_block(self, layers, ipt_dim):
        seq = []
        for l in layers:
            l_name, opt_dim, activation = l.split('_'); opt_dim = int(opt_dim)
            if opt_dim == -1: opt_dim = self.embed_dim
            if self.dropout: seq.append(self.dropout)
            if l_name == 'ff': seq.append(nn.Linear(ipt_dim, opt_dim))
            else: raise ValueError(f'MLPBlock l_name [{l_name}] not handled')
            # Layer activation
            if activation == 'rel': seq.append(nn.ReLU())
            elif activation == 'gel': seq.append(nn.GELU())
            elif activation == 'sig': seq.append(nn.Sigmoid())
            else: raise ValueError(f'MLPBlock activation [{activation}] not handled')
            ipt_dim = opt_dim
        return nn.Sequential(*seq)
        
    # x_input -> (#events, batch_size, emb_dim)
    # mask -> (batch_size, #events) -> (#events, batch_size)
    def forward(self, x_input, mask=None, pos_embs=None):   # , **args
        x = self.seq_init(x_input)
        if mask is not None: 
            mask = mask.reshape(mask.shape[1], mask.shape[0])
            
        if self.add_x_input: x = x + x_input
        
        return x
    
  
def get_block(name, params):
    if name == 'MLP': return MLPBlock(**params)
    elif name == 'TransformerBlock': return TransformerBlock(**params)
    else: raise ValueError(f'Block [{name}] not implemented')

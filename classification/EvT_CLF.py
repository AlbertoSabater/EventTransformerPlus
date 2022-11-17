from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import normal
import copy
import math


import sys
sys.path.append('..')
from model_utils.evt_utils import get_block, AttentionBlock, fourier_features, TransformerBlock



# Contains a set of latent memory vectors that are being updated with the information 
# processed at each time-step
class LatentMemory(nn.Module):
    def __init__(self, 
                 attention_params,
                 num_latent_vectors,
                 num_self_blocks=0,
                 ):
        super(LatentMemory, self).__init__()

        # To update latent vectors from event tokens
        self.cross_attention_layer_events = AttentionBlock(**attention_params, cross_att=True)

        self.num_self_blocks = num_self_blocks
        if num_self_blocks > 0:
            self.self_attention = TransformerBlock(latent_blocks=num_self_blocks-1, cross_heads=attention_params['heads'], num_latent_vectors=0, cross_att=False, **attention_params)
        
        opt_dim = attention_params['opt_dim']
        self.add_norm = nn.LayerNorm([opt_dim])

        self.latent_vectors = nn.Parameter(normal.Normal(0.0, 0.2).sample((num_latent_vectors, attention_params['opt_dim'])).clip(-2,2), requires_grad=True)


    # Erease memory from latent vectors and expand to batch_size
    def init_forward_pass(self, batch_size):
        # Reèat latent_vectors to batch_size
        self.prev_latent_vectors = self.latent_vectors.unsqueeze(1)
        self.prev_latent_vectors = self.prev_latent_vectors.expand(-1, batch_size, -1)    # (num_latent_vectors, batch_size, embed_dim)
        # Repeat expansion to keep as unaltered vetors for the token rollback
        self.raw_latent_vectors = self.latent_vectors.unsqueeze(1)
        self.raw_latent_vectors = self.raw_latent_vectors.expand(-1, batch_size, -1)    # (num_latent_vectors, batch_size, embed_dim)

    
    # Update latent memory vectors
    def update_latent_vectors_fn(self, z, mask_time_t):
        self.prev_latent_vectors = self.add_norm(self.prev_latent_vectors + z)
        
        # If any item from the processed batch do not have info -> do not update memory revert
        if mask_time_t is not None and mask_time_t.any(): 
            self.prev_latent_vectors[:, mask_time_t] = self.raw_latent_vectors[:, mask_time_t]
    

    # Process incoming token information and update latent vectors with each set of tokens
    def update_latent_vectors_from_tokens(self, event_tokens, mask_t=None, mask_time_t=None):
        # Update latent vectors
        new_latent_vectors_t = self.cross_attention_layer_events(event_tokens[:, ~mask_time_t], self.prev_latent_vectors[:, ~mask_time_t], mask=mask_t[~mask_time_t])
        if self.num_self_blocks > 0: 
            # Process new latent vectors with Self-Attention
            new_latent_vectors_t = self.self_attention(new_latent_vectors_t)
        new_latent_vectors = torch.zeros_like(self.prev_latent_vectors); new_latent_vectors[:,~mask_time_t] = new_latent_vectors_t

        self.update_latent_vectors_fn(new_latent_vectors, mask_time_t)
        
        
    
# Transforms a set of latent vectors/Q into a single descriptor
class CLF_Decoder(nn.Module):
    
    # Self-Attention + GAP + Linear
    def __init__(self, 
                 attention_params,
                 num_attention_blocks,
                 opt_dim, 
                 opt_classes, softmax):
        super(CLF_Decoder, self).__init__()
        
        self.num_attention_blocks = num_attention_blocks
        if num_attention_blocks > 0: 
            attention_params['opt_dim'] = opt_dim
            self.self_attention = TransformerBlock(latent_blocks=num_attention_blocks-1, cross_heads=attention_params['heads'], num_latent_vectors=0, cross_att=False, **attention_params)
        
        self.linear1 = nn.Linear(opt_dim, opt_dim)
        
        self.linear_2 = nn.Linear(opt_dim, opt_dim)
        self.linear_3 = nn.Linear(opt_dim, opt_classes)
        self.softmax = softmax
    
    # batch_size x num_latent x emb_dim
    def forward(self, z):
        if self.num_attention_blocks > 0: z = self.self_attention(z)[0]
        z = self.linear1(z)
        z = F.relu(z)
        # Average every latent
        z = z.mean(dim=0)
        z = F.relu(self.linear_2(z))
        clf = self.linear_3(z)
        if self.softmax: clf = F.log_softmax(clf, dim=1)
        return clf

    
class EvT_CLF(nn.Module):
    def __init__(self, 
                  embed_dim, events_dim,
                  downsample_pos_enc,  
                  pos_encoding, pos_enc_grad, 
                  token_projection,
                  preproc_tokens,
                  skip_conn_backbone,
                         
                  init_self_attention_params,
                  latent_memory_params,
                  decoder_params,
                 ):
        super(EvT_CLF, self).__init__()
    
        self.downsample_pos_enc = downsample_pos_enc
        self.embed_dim = embed_dim
        self.skip_conn_backbone = skip_conn_backbone

        pos_enc_params = copy.deepcopy(pos_encoding['params'])
        pos_enc_params['shape'] = (math.ceil(pos_encoding['params']['shape'][0]/downsample_pos_enc), math.ceil(pos_encoding['params']['shape'][1]/downsample_pos_enc))
        self.pos_encoding = nn.Parameter(fourier_features(**pos_enc_params).permute(1,2,0), requires_grad=pos_enc_grad)
        pos_emb_dim = self.pos_encoding.shape[2]
 
        # Event pre-proc block -> Linear transformation on tokens
        token_projection['params']['embed_dim'] = embed_dim

        # Events preprocessing -> Linear transformation on tokens
        preproc_tokens['params']['embed_dim'] = embed_dim

        self.projection_events = get_block(token_projection['name'], {**token_projection['params'], **{'ipt_dim': events_dim}})
        inp_dim = int(token_projection['params']['init_layers'][-1].split('_')[1])
        if inp_dim == -1: inp_dim = embed_dim
        self.preproc_block_events = get_block(preproc_tokens['name'], {**preproc_tokens['params'], **{'ipt_dim': inp_dim+pos_emb_dim}})
            
        init_self_attention_params['params']['opt_dim'] = embed_dim
        init_self_attention_params['params']['cross_att'] = False
        init_self_attention_params['params']['latent_blocks'] = init_self_attention_params['num_blocks']-1
        self.backbone_events = get_block(init_self_attention_params['name'], init_self_attention_params['params'])
        del init_self_attention_params['params']['cross_att']

        latent_memory_params['attention_params']['opt_dim'] = embed_dim
        self.cross_attention = LatentMemory(**latent_memory_params)
        
        decoder_params['opt_dim'] = embed_dim
        self.decoder = CLF_Decoder(**decoder_params)



    def forward(self, event_data):
        
        # Prepare data and get masks for empty tokens/time-steps
        tokens_events, pixels_events = event_data['event_frames']
        pixels_events = torch.div(pixels_events, self.downsample_pos_enc, rounding_mode='trunc')
        B, T, _, _= tokens_events.shape                  # (B, T, num_tokens, embed_dim)
        tokens_events = tokens_events.permute(1,0,2,3)    # (T, B, num_tokens, embed_dim)
        pixels_events = pixels_events.permute(1,0,2,3)    # (T, B, num_tokens, 2)
        samples_mask_events = tokens_events.sum(-1) == 0            #  to ignore in the attention block   
        samples_mask_time_events = tokens_events.sum(-1).sum(-1) == 0     # to ignore when there is no events at some time-step -> in a short clip when it is left-padded

        # Add tokens and preprocess
        tokens_events_1 = self.projection_events(tokens_events)              # (num_timesteps, batch_size, num_events, token_dim)
        pos_embs = self.pos_encoding[pixels_events[:,:,:,1], pixels_events[:,:,:,0],:]
        tokens_events_2 = torch.cat([tokens_events_1, pos_embs], dim=-1)
        tokens_events_2 = self.preproc_block_events(tokens_events_2)              # (num_timesteps, batch_size, token_dim, num_events)
        
        tokens_events = tokens_events_2
    
        # Erase memory from decoder latent vectors and initialize memory
        self.cross_attention.init_forward_pass(B)

        for num_time_step in range(T):
                                     
            mask_t = samples_mask_events[num_time_step]
            mask_time_t = samples_mask_time_events[num_time_step]
            shortcut = tokens_events[num_time_step].permute(1,0,2)         # (tokens, batch_size, emb_dim)
            
            if B == 1:
                # Remove empty tokens for a fair FLOP calculation
                shortcut = shortcut[~mask_t[0]]
                mask_t = mask_t[:,~mask_t[0]]
                if shortcut.shape[0] <= 1 or mask_time_t.all():
                    print(f'*** Skipping time_step {num_time_step} {mask_t.sum()} {B} {shortcut.shape}')
                    continue

            # Process tokens
            x = self.backbone_events(shortcut, mask=mask_t, mask_time_t=mask_time_t)[0]
            if self.skip_conn_backbone: x = x + shortcut
                
            # Update memory
            self.cross_attention.update_latent_vectors_from_tokens(x, mask_t=mask_t, mask_time_t=mask_time_t)
    
        return self.decoder(self.cross_attention.prev_latent_vectors)





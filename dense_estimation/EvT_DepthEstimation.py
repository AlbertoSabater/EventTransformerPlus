from torch import nn
import torch
from torch.distributions import normal

import sys
sys.path.append('..')
from model_utils.evt_utils import get_block, AttentionBlock, fourier_features, TransformerBlock

import copy
import math



# Contains a set of latent memory vectors that are updated with the new incoming information
# Latent vectors update processed tokens for the decoding step
class LatentMemory(nn.Module):
    def __init__(self, 
                 attention_params, num_latent_vectors, 
                 num_self_blocks,
                 use_events, use_images,
                 ):
        super(LatentMemory, self).__init__()
        
        self.use_events, self.use_images = use_events, use_images

        # To update latent vectors from event tokens
        self.cross_attention_layer_events = AttentionBlock(**attention_params, cross_att=True)
            
        # To update tokens from latent vectors
        self.cross_attention_layer_common = AttentionBlock(**attention_params, cross_att=True)
        self.num_self_blocks = num_self_blocks
        if num_self_blocks > 0: 
            self.self_attention = TransformerBlock(latent_blocks=num_self_blocks-1, cross_heads=attention_params['heads'], num_latent_vectors=0, cross_att=False, **attention_params)
        
        opt_dim = attention_params['opt_dim']
        self.add_norm = nn.LayerNorm([opt_dim])

        # Latent memory vectors
        self.latent_vectors = nn.Parameter(normal.Normal(0.0, 0.2).sample((num_latent_vectors, attention_params['opt_dim'])).clip(-2,2), requires_grad=True)

        inp_dim = opt_dim  
        self.new_z_info_norm = nn.LayerNorm([opt_dim])
        self.process_merge_tokens = get_block("MLP", {**{"init_layers": [f"ff_{opt_dim}_gel"], 'ipt_dim': inp_dim, 'embed_dim': opt_dim}})


    # Erease memory from latent vectors and set to batch_size
    def init_forward_pass(self, batch_size):
        # Reèat latent_vectors to batch_size
        self.prev_latent_vectors = self.latent_vectors.unsqueeze(1)
        self.prev_latent_vectors = self.prev_latent_vectors.expand(-1, batch_size, -1)    # (num_latent_vectors, batch_size, embed_dim)
        # Repeat expansion to keep as unaltered vetors for the token rollback
        self.raw_latent_vectors = self.latent_vectors.unsqueeze(1)
        self.raw_latent_vectors = self.raw_latent_vectors.expand(-1, batch_size, -1)    # (num_latent_vectors, batch_size, embed_dim)

    def update_latent_vectors_fn(self, z, mask_time_t):
        self.prev_latent_vectors = self.add_norm(self.prev_latent_vectors + z)
        
        if mask_time_t is not None and mask_time_t.any():       #  and self.update_latent_vectors_mode != 'raw' 
            self.prev_latent_vectors[:, mask_time_t] = self.raw_latent_vectors[:, mask_time_t]
    
    # Update latent vectors with each set of tokens
    # mask_time_t -> True to ignore
    def update_latent_vectors_from_tokens(self, event_tokens=None, mask_t=None, mask_time_t=None):      # , image_tokens=None
        new_latent_vectors_t = self.cross_attention_layer_events(event_tokens[:, ~mask_time_t], self.prev_latent_vectors[:, ~mask_time_t], mask=mask_t[~mask_time_t])
        if self.num_self_blocks > 0: 
            # Process new latent vectors with Self-Attention
            new_latent_vectors_t, _ = self.self_attention(new_latent_vectors_t)
        new_latent_vectors = torch.zeros_like(self.prev_latent_vectors); new_latent_vectors[:,~mask_time_t] = new_latent_vectors_t
        self.update_latent_vectors_fn(new_latent_vectors, mask_time_t)

    # Join (cooncat/add) the event and image tokens in a single Tensor
    def merge_tokens(self, event_tokens, image_tokens, B, H, W):
        if self.use_events and not self.use_images: return event_tokens
        if not self.use_events and self.use_images: return image_tokens
        # Add tokens
        common_tokens = self.new_z_info_norm(event_tokens + image_tokens)
        common_tokens = self.process_merge_tokens(common_tokens)
        return common_tokens

    # Update tokens with the info from the latent vectors
    def updaate_tokens_from_memory(self, common_tokens):
        if self.num_self_blocks > 0: lat_vecs, _ = self.self_attention(self.prev_latent_vectors)
        else: lat_vecs = self.prev_latent_vectors
        common_tokens = self.cross_attention_layer_common(lat_vecs, common_tokens)
        if common_tokens.isnan().any(): print('***** common_tokens NaN', common_tokens.isnan().sum())
        return common_tokens    
    
    
    
    
class DepthDecoder(nn.Module):
    def __init__(self, 
                   embed_dim, events_dim, images_dim,
                   downsample_pos_enc,  
                   pos_encoding, pos_enc_grad, 
                   token_projection,
                   preproc_tokens,
                         
                   init_self_attention_params,
                   latent_memory_params,
                   decoder_activation, 
                  
                   skip_blocks,
                   decoder_pos_enc,
                  
                   depth_size,
                   height, width,
                   patch_size,
                   use_events, use_images,
                   multimodal_drop,
                   skip_conn_backbone,
                 ):
        super(DepthDecoder, self).__init__()
        
        self.embed_dim, self.events_dim, self.images_dim = embed_dim, events_dim, images_dim,
        self.use_events, self.use_images = use_events, use_images
        self.downsample_pos_enc = downsample_pos_enc
        self.multimodal_drop = multimodal_drop
        self.depth_size = depth_size
        self.patch_size = patch_size
        self.skip_blocks = skip_blocks
        self.decoder_pos_enc = decoder_pos_enc
        self.decoder_activation = decoder_activation
        self.skip_conn_backbone = skip_conn_backbone
    
        self.orig_height, self.orig_width = height, width
        self.height, self.width = height, width
        # Downsample image shape
        # Make downsampled shape to be divisible by patch size
        self.diff_h , self.diff_w = self.height % patch_size, self.width % patch_size
        if self.diff_h > 0: self.diff_h = patch_size - self.diff_h
        if self.diff_w > 0: self.diff_w = patch_size - self.diff_w
        self.height += self.diff_h
        self.width += self.diff_w
        
        pos_enc_params = copy.deepcopy(pos_encoding['params'])
        pos_enc_params['shape'] = (math.ceil(pos_encoding['params']['shape'][0]/downsample_pos_enc), math.ceil(pos_encoding['params']['shape'][1]/downsample_pos_enc))
        self.pos_encoding = nn.Parameter(fourier_features(**pos_enc_params).permute(1,2,0), requires_grad=pos_enc_grad)
        pos_emb_dim = self.pos_encoding.shape[2]
        

        # Event pre-proc block -> Linear transformation on tokens
        token_projection['params']['embed_dim'] = embed_dim

        # Events preprocessing -> Linear transformation on tokens
        preproc_tokens['params']['embed_dim'] = embed_dim
        
        # Tokens self-attention pre-processing
        init_self_attention_params['opt_dim'] = embed_dim
        if use_events: 
            self.projection_events = get_block(token_projection['name'], {**token_projection['params'], **{'ipt_dim': events_dim}})
            inp_dim = int(token_projection['params']['init_layers'][-1].split('_')[1])
            if inp_dim == -1: inp_dim = embed_dim
            self.preproc_block_events = get_block(preproc_tokens['name'], {**preproc_tokens['params'], **{'ipt_dim': inp_dim+pos_emb_dim}})
            self.backbone_events = TransformerBlock(**init_self_attention_params, cross_att=False, return_intermediate_results=skip_blocks)
        if use_images: 
            self.projection_images = get_block(token_projection['name'], {**token_projection['params'], **{'ipt_dim': images_dim}})
            inp_dim = int(token_projection['params']['init_layers'][-1].split('_')[1])
            if inp_dim == -1: inp_dim = embed_dim
            self.preproc_block_images = get_block(preproc_tokens['name'], {**preproc_tokens['params'], **{'ipt_dim': inp_dim+pos_emb_dim}})
            self.backbone_images = TransformerBlock(**init_self_attention_params, cross_att=False, return_intermediate_results=skip_blocks)


        # Latent memory
        latent_memory_params['attention_params']['opt_dim'] = embed_dim
        latent_memory_params['use_events'] = use_events
        latent_memory_params['use_images'] = use_images
        self.cross_attention = LatentMemory(**latent_memory_params)     # , single_vectors_update=single_vectors_update

        if decoder_pos_enc:
            self.ff_decoder_pos_enc = get_block('MLP', {**{"init_layers": [f"ff_{embed_dim}_gel"], 'ipt_dim': embed_dim+pos_emb_dim, 'embed_dim': embed_dim}})
    
        self.norm_merge_skip_tokens = nn.LayerNorm([embed_dim])

        # Decoder transformer
        init_self_attention_params['opt_dim_new'] = embed_dim*2 #  if decoder['params']['use_prev_blocks_out_params']['mode'] == 'concat' else backbone_params['embed_dim']
        init_self_attention_params['latent_blocks'] += 1
        self.decoder = TransformerBlock(**init_self_attention_params, cross_att=False, get_intermediate_results=skip_blocks)
        self.decoder_norm = nn.LayerNorm([embed_dim])
        # Linear layer to set each token (latent_vector) of size (decoder_patch_size x decoder_patch_size)
        if decoder_activation == 'relu': act = 'rel' 
        elif decoder_activation == 'sigmoid': act = 'sig'
        else: raise ValueError('decoder activation [{decoder_activation}] not handled')
        self.decoder_out = get_block("MLP", {**{"init_layers": [f"ff_{patch_size*patch_size}_{act}"], 'ipt_dim': embed_dim, 'embed_dim': embed_dim}})


    def sparse_tokens_to_dense(self, tokens, pixels, B):
        # Build dense representation from dense tokens
        height, width = self.height//self.patch_size, self.width//self.patch_size
        new_info = torch.zeros(B * height * width, self.embed_dim, device=tokens.device)
        tokens = tokens.permute(1, 0, 2)  # (B, num_tokens, dim)
        num_tokens = pixels.shape[1]
        prev_pixels_trans = pixels[:,:,0] * width + pixels[:,:,1]
        prev_pixels_trans = (torch.arange(B, device=tokens.device)*(height*width)).repeat_interleave(num_tokens).reshape(B, num_tokens) + prev_pixels_trans
        prev_pixels_trans = prev_pixels_trans.view(B * num_tokens)
        # Assign flatten tokens to new_info
        new_info[prev_pixels_trans] = tokens.contiguous().view(B*num_tokens, self.embed_dim)
        # flattened new_info to batched pt
        new_info = new_info.contiguous().view(B, -1, self.embed_dim).permute(1,0,2)
        return new_info
    
    # Merge tokens for the later skip-connection. Add * norm
    def merge_inter_res(self, inter_res_events, inter_res_images):
        inter_res = { k:self.norm_merge_skip_tokens(inter_res_events[k] + inter_res_images[k]) for k in inter_res_events }
        return inter_res
    
    # Decoding
    def decode_latent_vectors(self, B, z, inter_res=None):
        # Output processed decoder latent vectors
        z, _ = self.decoder(z, prev_inter_tokens=inter_res)
        # Transfor latent vectors to the decoder patch_size
        z = self.decoder_norm(z)
        z = self.decoder_out(z)
        # Reconstruct original shape
        z = z.permute(1, 0, 2).contiguous().view(B, self.height, self.width)
        # Upsample
        z = z[:,None,...]
        return z    
    
    
    def forward(self, event_data):
        # =============================================================================
        # Pre-processing
        # =============================================================================
        # Prepare multimodal tokens
        # Downsample pixels -> add pos. enc. -> Project
        if self.use_events: 
            tokens_events, pixels_events = event_data['event_frames']
            pixels_events = torch.div(pixels_events, self.downsample_pos_enc, rounding_mode='trunc')
            B, T, _, _= tokens_events.shape                  # (B, T, num_tokens, embed_dim)
            tokens_events = tokens_events.permute(1,0,2,3)    # (T, B, num_tokens, embed_dim)
            pixels_events = pixels_events.permute(1,0,2,3)    # (T, B, num_tokens, 2)
            samples_mask_events = tokens_events.sum(-1) == 0            #  to ignore in the attention block   
            samples_mask_time_events = tokens_events.sum(-1).sum(-1) == 0     # to ignore when there is no events at some time-step -> in a short clip when it is left-padded
            
            tokens_events = self.projection_events(tokens_events)              # (num_timesteps, batch_size, num_events, token_dim)
            pos_embs = self.pos_encoding[pixels_events[:,:,:,0], pixels_events[:,:,:,1],:]
            tokens_events = torch.cat([tokens_events, pos_embs], dim=-1)
            tokens_events = self.preproc_block_events(tokens_events)              # (num_timesteps, batch_size, token_dim, num_events)

            pixels_events = [ pi for pi in pixels_events ]
            
        if self.use_images: 
            tokens_images, pixels_images = event_data['images']
            pixels_images = torch.div(pixels_images, self.downsample_pos_enc, rounding_mode='trunc')
            B, T, _, _= tokens_images.shape
            tokens_images = tokens_images.permute(1,0,2,3)
            pixels_images = pixels_images.permute(1,0,2,3)
            samples_mask_images = tokens_images.sum(-1) == 0            #  to ignore in the attention block   
            samples_mask_time_images = tokens_images.sum(-1).sum(-1) == 0     # to ignore when there is no events at some time-step -> in a short clip when it is left-padded
            
            # Add tokens and preprocess
            tokens_images = self.projection_images(tokens_images)              # (num_timesteps, batch_size, num_events, token_dim)
            pos_embs = self.pos_encoding[pixels_images[:,:,:,0], pixels_images[:,:,:,1],:]
            tokens_images = torch.cat([tokens_images, pos_embs], dim=-1)
            tokens_images = self.preproc_block_images(tokens_images)              # (num_timesteps, batch_size, token_dim, num_events)
            
        
        # Erase memory from decoder latent vectors 
        self.cross_attention.init_forward_pass(B)
        
        if self.decoder_pos_enc:
            pixels_h = torch.arange(0, self.height, self.patch_size).repeat_interleave(self.width//self.patch_size)
            pixels_w = torch.arange(0, self.width, self.patch_size).repeat(self.height//self.patch_size)   
            decoding_pixels = torch.stack([pixels_h, pixels_w], axis=-1).to(device=tokens_images.device if self.use_images else tokens_events.device)
            decoding_pixels = decoding_pixels.expand(B,*decoding_pixels.shape)
            decoding_pixels = torch.div(decoding_pixels, self.downsample_pos_enc, rounding_mode='trunc')
            
        # Set multimodal drop params
        # If training in multimodal -> randomly not use either events or images
        drop_events, drop_images = False, False
        if self.multimodal_drop and self.training and self.use_events and self.use_images:
            rnd = torch.rand(1)
            if rnd < 0.3:   drop_events, drop_images = True, False
            elif rnd < 0.6: drop_events, drop_images = False, True
            else:           drop_events, drop_images = False, False
            
        
        res_images, mask_t_img = torch.zeros((0,B,self.embed_dim), device=self.pos_encoding.device), torch.zeros((0), device=self.pos_encoding.device)
        res_events, mask_t_evn = torch.zeros((0,B,self.embed_dim), device=self.pos_encoding.device), torch.zeros((0), device=self.pos_encoding.device)
        inter_res, inter_res_events, inter_res_images = None, None, None
        inter_res_events_dense = { k: torch.zeros((self.height * self.width // self.patch_size**2, B, self.embed_dim), device=self.pos_encoding.device) for k in range(self.backbone_events.latent_blocks+1) }
        inter_res_images_dense = { k: torch.zeros((self.height * self.width // self.patch_size**2, B, self.embed_dim), device=self.pos_encoding.device) for k in range(self.backbone_events.latent_blocks+1) }
        res_events_dense = torch.zeros((self.height * self.width // self.patch_size**2, B, self.embed_dim), device=self.pos_encoding.device)
        res_images_dense = torch.zeros((self.height * self.width // self.patch_size**2, B, self.embed_dim), device=self.pos_encoding.device)
        decodings = []
        for t in range(T):
            
            """
            If one time-step does not have event/image information we take the one from the previous time-step
                for the decoding
            """
            # =============================================================================
            # Backbone processing
            # =============================================================================
            mask_time_t_img, mask_time_t_evn = torch.zeros((B), device = self.pos_encoding.device), torch.zeros((B), device = self.pos_encoding.device)
            
            # Token Self-Attention -> update memory
            if self.use_images: 
                if not drop_images and not samples_mask_time_images[t].all():   
                    mask_t_img = samples_mask_images[t]
                    mask_time_t_img = samples_mask_time_images[t]
                    shortcut_img = tokens_images[t].permute(1,0,2)     
                    pixels_images_t = pixels_images[t]                      # (B, num_tokens, 2))
                    res_images, inter_res_images = self.backbone_images(shortcut_img, mask=mask_t_img, mask_time_t=mask_time_t_img)
                    if self.skip_conn_backbone: res_images = res_images + shortcut_img

            if self.use_events: 
                if not drop_events and not samples_mask_time_events[t].all():
                    mask_t_evn = samples_mask_events[t]
                    mask_time_t_evn = samples_mask_time_events[t]
                    shortcut_evn = tokens_events[t].permute(1,0,2)  # (num_tokens, B, dim)
                    pixels_events_t = pixels_events[t]                      # (B, num_tokens, 2))
                    res_events, inter_res_events = self.backbone_events(shortcut_evn, mask=mask_t_evn, mask_time_t=mask_time_t_evn)
                    if self.skip_conn_backbone: res_events = res_events + shortcut_evn
                    
            # =============================================================================
            #  Memory update
            # =============================================================================
            sys.stdout.flush()
            tkns = torch.cat([res_images, res_events])                  # (num_tokens, B, dim)
            mask_t = torch.cat([mask_t_img, mask_t_evn], axis=1)        # (B, num_tokens)
            mask_time_t = (mask_time_t_img + mask_time_t_evn) != 0      # (B)
            if not mask_time_t.all() and tkns.shape[0] > 0:
                self.cross_attention.update_latent_vectors_from_tokens(tkns, mask_t=mask_t, mask_time_t=mask_time_t)
                
            # =============================================================================
            # Decoding
            # =============================================================================
            if T-t <= self.depth_size:
                if self.use_events: 
                    if self.skip_blocks and inter_res_events is not None: 
                        inter_res_events_dense = { k:self.sparse_tokens_to_dense(inter_res_events[k], pixels_events_t, B) for k in inter_res_events.keys() }
                    if res_events.sum() != 0: res_events_dense = self.sparse_tokens_to_dense(res_events, pixels_events_t, B)
                if self.use_images: 
                    if self.skip_blocks and inter_res_images is not None: 
                        inter_res_images_dense = { k:self.sparse_tokens_to_dense(inter_res_images[k], pixels_images_t, B) for k in inter_res_images.keys() }
                    if res_images.sum() != 0: res_images_dense = self.sparse_tokens_to_dense(res_images, pixels_images[t], B)
                
                # Merge image and event tokens
                common_tokens = self.cross_attention.merge_tokens(res_events_dense, res_images_dense, B, self.height, self.width)
                
                # Add positional information
                if self.decoder_pos_enc:
                    common_tokens = torch.cat([common_tokens, 
                            self.pos_encoding[decoding_pixels[:,:,0], decoding_pixels[:,:,1],:].permute(1,0,2)], dim=-1)
                    common_tokens = self.ff_decoder_pos_enc(common_tokens)
                
                # Update tokens with info from the latent memory
                common_tokens = self.cross_attention.updaate_tokens_from_memory(common_tokens)
                
                if self.skip_blocks: inter_res = self.merge_inter_res(inter_res_events_dense, inter_res_images_dense)
                decodings.append(self.decode_latent_vectors(B, common_tokens, inter_res))
                
        return torch.cat(decodings, axis=1)
        
    
    
    
            
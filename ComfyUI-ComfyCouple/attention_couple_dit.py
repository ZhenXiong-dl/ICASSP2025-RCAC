import torch
import torch.nn.functional as F
import copy
import comfy
from comfy.ldm.modules.attention import optimized_attention

def get_masks_from_q(masks, q, original_shape):

    if original_shape[2] * original_shape[3] == q.shape[1]:
        down_sample_rate = 1
    elif (original_shape[2] // 2) * (original_shape[3] // 2) == q.shape[1]:
        down_sample_rate = 2
    elif (original_shape[2] // 4) * (original_shape[3] // 4) == q.shape[1]:
        down_sample_rate = 4
    else:
        down_sample_rate = 8

    ret_masks = []
    for mask in masks:
        if isinstance(mask,torch.Tensor):
            size = (original_shape[2] // down_sample_rate, original_shape[3] // down_sample_rate)
            mask_downsample = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest")
            mask_downsample = mask_downsample.view(1,-1, 1).repeat(q.shape[0], 1, q.shape[2])
            ret_masks.append(mask_downsample)
        else: 
            ret_masks.append(torch.ones_like(q))
    
    ret_masks = torch.cat(ret_masks, dim=0)
    return ret_masks

def set_model_patch_replace(model, patch, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    to["patches_replace"]["attn2"][key] = patch

class AttentionCouple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["Attention", "Latent"], ),
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive, negative, mode):
        if mode == "Latent":
            return (model, positive, negative)                       #  直接的lanten组合
        
        self.negative_positive_masks = []
        self.negative_positive_conds = []
        self.negative_positive_conds_t5 = []
        
        new_positive = copy.deepcopy(positive)
        new_negative = copy.deepcopy(negative)
        
        dtype = model.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()
        
        for conditions in [new_negative, new_positive]:
            conditions_masks = []
            conditions_conds = []
            conditions_conds_t5 = []
            if len(conditions) != 1:         # conditions === positive  (len=2)
                mask_norm = torch.stack([cond[1]["mask"].to(device, dtype=dtype) * cond[1]["mask_strength"] for cond in conditions])
                mask_norm = mask_norm / mask_norm.sum(dim=0)                                 # mask归一化
                conditions_masks.extend([mask_norm[i] for i in range(mask_norm.shape[0])])   #
                conditions_conds.extend([cond[0].to(device, dtype=dtype) for cond in conditions])
                conditions_conds_t5.extend([cond[1] for cond in conditions])
                # conditions_conds.extend([[cond[0].to(device, dtype=dtype),cond[1]] for cond in conditions])     # add mt5xl, add attention_mask by cond[1]
                del conditions[0][1]["mask"]                                                 # latent couple delete
                del conditions[0][1]["mask_strength"]
            else:
                conditions_masks = [False]
                # conditions_conds = [conditions[0][0].to(device, dtype=dtype),conditions[0][1]]
                conditions_conds = [conditions[0][0].to(device, dtype=dtype)]
                conditions_conds_t5 = [conditions[0][1]]
            self.negative_positive_masks.append(conditions_masks)
            self.negative_positive_conds.append(conditions_conds)
            self.negative_positive_conds_t5.append(conditions_conds_t5)
        self.conditioning_length = (len(new_negative), len(new_positive))

        new_model = model.clone()
        self.sdxl = hasattr(new_model.model.diffusion_model, "label_emb")
        hydit = hasattr(new_model.model.diffusion_model, "mlp_t5")
        if self.sdxl:      # SDXL
            for id in [4,5,7,8]:                                                         # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10)                   # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[index].attn2), ("input", id, index))
            for index in range(10):
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[index].attn2), ("middle", id, index))
            for id in range(6):                                                          # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10)               # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[index].attn2), ("output", id, index))
        elif hydit:
            for index in range(40):
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.blocks[index].attn2), ("blocks",index))
        else:       # SD1.5
            for id in [1,2,4,5,7,8]:                                                     # id of input_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[0].attn2), ("input", id))
            set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2), ("middle", 0))
            for id in [3,4,5,6,7,8,9,10,11]:                                             # id of output_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[0].attn2), ("output", id))

        return (new_model, [new_positive[0]], [new_negative[0]])                         # pool output

    def make_patch(self, module):
        def patch(q, k, v, extra_options):
            mlp_t5=extra_options['mlp_t5']
            
            len_neg, len_pos = self.conditioning_length                # negative, positive的长度
            cond_or_uncond = extra_options["cond_or_uncond"]             # 0: cond, 1: uncond
            q_list = q.chunk(len(cond_or_uncond), dim=0)
            b = q_list[0].shape[0] # batch_size
                
            # set mask for q
            masks_uncond = get_masks_from_q(self.negative_positive_masks[0], q_list[0], extra_options["original_shape"])
            masks_cond = get_masks_from_q(self.negative_positive_masks[1], q_list[0], extra_options["original_shape"])

            # context_uncond = torch.cat([cond for cond in self.negative_positive_conds[0]], dim=0)  # old
            # context_cond = torch.cat([cond for cond in self.negative_positive_conds[1]], dim=0)
            
            # project negativae prompts
            ## old:  context_uncond= self.negative_positive_conds[0][0]   
            device = comfy.model_management.get_torch_device()
            dtype =extra_options['dtype']
            self.text_len=extra_options['text_len']
            self.text_len_t5=extra_options['text_len_t5']
            #type = model.model.diffusion_model.dtype
            
            # project negative
            text_states = self.negative_positive_conds[0][0].to(device,dtype=dtype)                                            # 2,77,1024   clip
            text_states_t5 = self.negative_positive_conds_t5[0][0]['conditioning_mt5xl'].to(device,dtype=dtype)                    # 2,256,2048[]
            text_states_mask = self.negative_positive_conds_t5[0][0]['attention_mask'].bool().to(device)                # 2,77   text_embedding_mask.bool()
            text_states_t5_mask = self.negative_positive_conds_t5[0][0]['attention_mask_mt5xl'].bool().to(device)       # 2,256  text_embedding_mask_t5.bool()
            b_t5, l_t5, c_t5 = text_states_t5.shape
            text_states_t5 = mlp_t5(text_states_t5.view(-1, c_t5)).view(b_t5, l_t5, -1)
            padding = comfy.ops.cast_to_input(extra_options['text_embedding_padding'], text_states)
            text_states[:,-self.text_len:] = torch.where(text_states_mask[:,-self.text_len:].unsqueeze(2), text_states[:,-self.text_len:], padding[:self.text_len])
            text_states_t5[:,-self.text_len_t5:] = torch.where(text_states_t5_mask[:,-self.text_len_t5:].unsqueeze(2), text_states_t5[:,-self.text_len_t5:], padding[self.text_len:])
            text_states = torch.cat([text_states, text_states_t5], dim=1)  # 2,205，1024    
            context_uncond=text_states # [1,333,1024]
            
            # project  all positive prompts 
            all_states=[]
            for mask_index in range(len_pos):
                text_states = self.negative_positive_conds[1][mask_index].to(device,dtype=dtype)                                          # 2,77,1024   clip
                text_states_t5 = self.negative_positive_conds_t5[1][mask_index]['conditioning_mt5xl'].to(device,dtype=dtype)                # 2,256,2048[]
                text_states_mask = self.negative_positive_conds_t5[1][mask_index]['attention_mask'].bool().to(device)              # 2,77   text_embedding_mask.bool()
                text_states_t5_mask = self.negative_positive_conds_t5[1][mask_index]['attention_mask_mt5xl'].bool().to(device)     # 2,256  text_embedding_mask_t5.bool()
                b_t5, l_t5, c_t5 = text_states_t5.shape
                text_states_t5 = mlp_t5(text_states_t5.view(-1, c_t5)).view(b_t5, l_t5, -1)
                padding = comfy.ops.cast_to_input(extra_options['text_embedding_padding'], text_states)
                text_states[:,-self.text_len:] = torch.where(text_states_mask[:,-self.text_len:].unsqueeze(2), text_states[:,-self.text_len:], padding[:self.text_len])
                text_states_t5[:,-self.text_len_t5:] = torch.where(text_states_t5_mask[:,-self.text_len_t5:].unsqueeze(2), text_states_t5[:,-self.text_len_t5:], padding[self.text_len:])
                text_states = torch.cat([text_states, text_states_t5], dim=1)  # 2,205，1024    
                all_states.append(text_states)
            context_cond= torch.cat(all_states, dim=0)
            
            s2=context_uncond.shape[1]
            kv_uncond  = module.kv_proj(context_uncond).view(b, s2, 2, module.num_heads* module.head_dim)        # [b, s2, 2, h* d]
            # kv_uncond  = module.kv_proj(context_uncond).view(b, s2, 2, module.num_heads, module.head_dim)    # [b, s2, 2, h, d]
            k_uncond, v_uncond= kv_uncond.unbind(dim=2) # [b, s, h, d]
            s2=context_cond.shape[1]
            kv_cond  = module.kv_proj(context_cond).view(b*len(context_cond), s2, 2, module.num_heads*module.head_dim)
            # kv_cond  = module.kv_proj(context_cond).view(b*len(context_cond), s2, 2, module.num_heads, module.head_dim)    # [b, s2, 2, h, d]
            k_cond, v_cond= kv_cond.unbind(dim=2) # [b, s, h, d]
            k_cond = module.k_norm(k_cond.view(b*len(context_cond), s2, module.num_heads,module.head_dim)).view(b*len(context_cond), s2, module.num_heads*module.head_dim)
            k_uncond = module.k_norm(k_uncond.view(b*len(context_uncond), s2, module.num_heads,module.head_dim)).view(b*len(context_uncond), s2, module.num_heads*module.head_dim)
            
            out = []
            for i, c in enumerate(cond_or_uncond):
                if c == 0:   # 0: cond
                    masks = masks_cond
                    k = k_cond
                    v = v_cond
                    length = len_pos
                else:
                    masks = masks_uncond
                    k = k_uncond
                    v = v_uncond
                    length = len_neg

                q_target = q_list[i].repeat(length, 1, 1)
                k = torch.cat([k[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)
                v = torch.cat([v[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)
                
                if k.dtype != q_target.dtype or v.dtype != q_target.dtype:
                    # Ensure all dtypes match
                    k = k.to(q_target.dtype)
                    v = v.to(q_target.dtype)
                qkv = optimized_attention(q_target, k, v, extra_options["n_heads"])
                # context = optimized_attention(q, k, v, self.num_heads, skip_reshape=True, attn_precision=self.attn_precision)
                
                qkv = qkv * masks
                qkv = qkv.view(length, b, -1, module.num_heads * module.head_dim).sum(dim=0)  # add positive promts qkv together
                qkv = module.out_proj(qkv) 
                qkv=module.proj_drop(qkv)
                out.append(qkv)

            out = torch.cat(out, dim=0)

            return out
        return patch

class AttentionCoupleMiddle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["Attention", "Latent"], ),
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive,positive_all, negative, block_nums, mode):
        if mode == "Latent":
            return (model, positive, negative)                       #  直接的lanten组合
        
        self.negative_positive_masks = []
        self.negative_positive_conds = []
        self.negative_positive_conds_t5 = []
        
        new_positive = copy.deepcopy(positive)
        new_positive_all = copy.deepcopy(positive_all)
        new_negative = copy.deepcopy(negative)
        
        
        dtype = model.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()
        
        for conditions in [new_negative, new_positive]:
            conditions_masks = []
            conditions_conds = []
            conditions_conds_t5 = []
            if len(conditions) != 1:         # conditions === positive  (len=2)
                mask_norm = torch.stack([cond[1]["mask"].to(device, dtype=dtype) * cond[1]["mask_strength"] for cond in conditions])
                mask_norm = mask_norm / mask_norm.sum(dim=0)                                 # mask归一化
                conditions_masks.extend([mask_norm[i] for i in range(mask_norm.shape[0])])   #
                conditions_conds.extend([cond[0].to(device, dtype=dtype) for cond in conditions])
                conditions_conds_t5.extend([cond[1] for cond in conditions])
                # conditions_conds.extend([[cond[0].to(device, dtype=dtype),cond[1]] for cond in conditions])     # add mt5xl, add attention_mask by cond[1]
                del conditions[0][1]["mask"]                                                 # latent couple delete
                del conditions[0][1]["mask_strength"]
            else:
                conditions_masks = [False]
                # conditions_conds = [conditions[0][0].to(device, dtype=dtype),conditions[0][1]]
                conditions_conds = [conditions[0][0].to(device, dtype=dtype)]
                conditions_conds_t5 = [conditions[0][1]]
            self.negative_positive_masks.append(conditions_masks)
            self.negative_positive_conds.append(conditions_conds)
            self.negative_positive_conds_t5.append(conditions_conds_t5)
        self.conditioning_length = (len(new_negative), len(new_positive))

        new_model = model.clone()
        self.sdxl = hasattr(new_model.model.diffusion_model, "label_emb")
        hydit = hasattr(new_model.model.diffusion_model, "mlp_t5")
        if self.sdxl:      # SDXL
            for id in [4,5,7,8]:                                                         # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10)                   # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[index].attn2), ("input", id, index))
            for index in range(10):
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[index].attn2), ("middle", id, index))
            for id in range(6):                                                          # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10)               # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[index].attn2), ("output", id, index))
        elif hydit:
            for index in range(block_nums):            # set for block_nums=21 x blocks in firts
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.blocks[index].attn2), ("blocks",index))
        else:       # SD1.5
            for id in [1,2,4,5,7,8]:                                                     # id of input_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[0].attn2), ("input", id))
            set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2), ("middle", 0))
            for id in [3,4,5,6,7,8,9,10,11]:                                             # id of output_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[0].attn2), ("output", id))

        return (new_model, [new_positive_all[0]], [new_negative[0]])       
                      # pool output
    def make_patch(self, module):
        def patch(q, k, v, extra_options):
            mlp_t5=extra_options['mlp_t5']
            
            len_neg, len_pos = self.conditioning_length                # negative, positive的长度
            cond_or_uncond = extra_options["cond_or_uncond"]             # 0: cond, 1: uncond
            q_list = q.chunk(len(cond_or_uncond), dim=0)
            b = q_list[0].shape[0] # batch_size
                
            # set mask for q
            masks_uncond = get_masks_from_q(self.negative_positive_masks[0], q_list[0], extra_options["original_shape"])
            masks_cond = get_masks_from_q(self.negative_positive_masks[1], q_list[0], extra_options["original_shape"])

            device = comfy.model_management.get_torch_device()
            dtype =extra_options['dtype']
            self.text_len=extra_options['text_len']
            self.text_len_t5=extra_options['text_len_t5']
            
            # project negative
            text_states = self.negative_positive_conds[0][0].to(device,dtype=dtype)                                            # 2,77,1024   clip
            text_states_t5 = self.negative_positive_conds_t5[0][0]['conditioning_mt5xl'].to(device,dtype=dtype)                    # 2,256,2048[]
            text_states_mask = self.negative_positive_conds_t5[0][0]['attention_mask'].bool().to(device)                # 2,77   text_embedding_mask.bool()
            text_states_t5_mask = self.negative_positive_conds_t5[0][0]['attention_mask_mt5xl'].bool().to(device)       # 2,256  text_embedding_mask_t5.bool()
            b_t5, l_t5, c_t5 = text_states_t5.shape
            text_states_t5 = mlp_t5(text_states_t5.view(-1, c_t5)).view(b_t5, l_t5, -1)
            padding = comfy.ops.cast_to_input(extra_options['text_embedding_padding'], text_states)
            text_states[:,-self.text_len:] = torch.where(text_states_mask[:,-self.text_len:].unsqueeze(2), text_states[:,-self.text_len:], padding[:self.text_len])
            text_states_t5[:,-self.text_len_t5:] = torch.where(text_states_t5_mask[:,-self.text_len_t5:].unsqueeze(2), text_states_t5[:,-self.text_len_t5:], padding[self.text_len:])
            text_states = torch.cat([text_states, text_states_t5], dim=1)  # 2,205，1024    
            context_uncond=text_states # [1,333,1024]
            
            # project  all positive prompts 
            all_states=[]
            for mask_index in range(len_pos):
                text_states = self.negative_positive_conds[1][mask_index].to(device,dtype=dtype)                                          # 2,77,1024   clip
                text_states_t5 = self.negative_positive_conds_t5[1][mask_index]['conditioning_mt5xl'].to(device,dtype=dtype)                # 2,256,2048[]
                text_states_mask = self.negative_positive_conds_t5[1][mask_index]['attention_mask'].bool().to(device)              # 2,77   text_embedding_mask.bool()
                text_states_t5_mask = self.negative_positive_conds_t5[1][mask_index]['attention_mask_mt5xl'].bool().to(device)     # 2,256  text_embedding_mask_t5.bool()
                b_t5, l_t5, c_t5 = text_states_t5.shape
                text_states_t5 = mlp_t5(text_states_t5.view(-1, c_t5)).view(b_t5, l_t5, -1)
                padding = comfy.ops.cast_to_input(extra_options['text_embedding_padding'], text_states)
                text_states[:,-self.text_len:] = torch.where(text_states_mask[:,-self.text_len:].unsqueeze(2), text_states[:,-self.text_len:], padding[:self.text_len])
                text_states_t5[:,-self.text_len_t5:] = torch.where(text_states_t5_mask[:,-self.text_len_t5:].unsqueeze(2), text_states_t5[:,-self.text_len_t5:], padding[self.text_len:])
                text_states = torch.cat([text_states, text_states_t5], dim=1)  # 2,205，1024    
                all_states.append(text_states)
            context_cond= torch.cat(all_states, dim=0)
            
            s2=context_uncond.shape[1]
            kv_uncond  = module.kv_proj(context_uncond).view(b, s2, 2, module.num_heads* module.head_dim)        # [b, s2, 2, h* d]
            # kv_uncond  = module.kv_proj(context_uncond).view(b, s2, 2, module.num_heads, module.head_dim)    # [b, s2, 2, h, d]
            k_uncond, v_uncond= kv_uncond.unbind(dim=2) # [b, s, h, d]
            s2=context_cond.shape[1]
            kv_cond  = module.kv_proj(context_cond).view(b*len(context_cond), s2, 2, module.num_heads*module.head_dim)
            # kv_cond  = module.kv_proj(context_cond).view(b*len(context_cond), s2, 2, module.num_heads, module.head_dim)    # [b, s2, 2, h, d]
            k_cond, v_cond= kv_cond.unbind(dim=2) # [b, s, h, d]
            k_cond = module.k_norm(k_cond.view(b*len(context_cond), s2, module.num_heads,module.head_dim)).view(b*len(context_cond), s2, module.num_heads*module.head_dim)
            k_uncond = module.k_norm(k_uncond.view(b*len(context_uncond), s2, module.num_heads,module.head_dim)).view(b*len(context_uncond), s2, module.num_heads*module.head_dim)
            
            out = []
            for i, c in enumerate(cond_or_uncond):
                if c == 0:   # 0: cond
                    masks = masks_cond
                    k = k_cond
                    v = v_cond
                    length = len_pos
                else:
                    masks = masks_uncond
                    k = k_uncond
                    v = v_uncond
                    length = len_neg

                q_target = q_list[i].repeat(length, 1, 1)
                k = torch.cat([k[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)
                v = torch.cat([v[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)
                
                if k.dtype != q_target.dtype or v.dtype != q_target.dtype:
                    # Ensure all dtypes match
                    k = k.to(q_target.dtype)
                    v = v.to(q_target.dtype)
                qkv = optimized_attention(q_target, k, v, extra_options["n_heads"])
                # context = optimized_attention(q, k, v, self.num_heads, skip_reshape=True, attn_precision=self.attn_precision)
                
                qkv = qkv * masks
                qkv = qkv.view(length, b, -1, module.num_heads * module.head_dim).sum(dim=0)  # add positive promts qkv together
                qkv = module.out_proj(qkv) 
                qkv=module.proj_drop(qkv)
                out.append(qkv)

            out = torch.cat(out, dim=0)

            return out
        return patch

NODE_CLASS_MAPPINGS = {
    "Attention couple": AttentionCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Attention couple": "Load Attention couple",
}

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.math import generate_permute_matrix
from utils.image import one_hot_mask

from networks.layers.basic import seq_to_2d


class DMAOTEngine(nn.Module):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 max_len_long_term=9999):
        super().__init__()

        self.cfg = aot_model.cfg
        self.align_corners = aot_model.cfg.MODEL_ALIGN_CORNERS
        self.AOT = aot_model

        self.max_obj_num = aot_model.max_obj_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.max_len_long_term = max_len_long_term
        self.losses = None

        self.restart_engine()

    def forward(self,
                all_frames,
                all_masks,
                batch_size,
                obj_nums,
                step=0,
                tf_board=False,
                use_prev_pred=False,
                enable_prev_frame=False,
                use_prev_prob=False):  # only used for training
        if self.losses is None:
            self._init_losses()

        self.freeze_id = True if use_prev_pred else False
        aux_weight = self.aux_weight * max(self.aux_step - step,
                                           0.) / self.aux_step

        self.offline_encoder(all_frames, all_masks)

        self.add_reference_frame(frame_step=0, obj_nums=obj_nums)

        grad_state = torch.no_grad if aux_weight == 0 else torch.enable_grad
        with grad_state():
            ref_aux_loss, ref_aux_mask = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step)

        aux_losses = [ref_aux_loss]
        aux_masks = [ref_aux_mask]

        curr_losses, curr_masks = [], []
        if enable_prev_frame:
            self.set_prev_frame(frame_step=1)
            with grad_state():
                prev_aux_loss, prev_aux_mask = self.generate_loss_mask(
                    self.offline_masks[self.frame_step], step)
            aux_losses.append(prev_aux_loss)
            aux_masks.append(prev_aux_mask)
        else:
            self.match_propogate_one_frame()
            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step, return_prob=True)
            self.update_short_term_memory(
                curr_mask if not use_prev_prob else curr_prob,
                None if use_prev_pred else self.assign_identity(
                    self.offline_one_hot_masks[self.frame_step]))
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)

        self.match_propogate_one_frame()
        curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
            self.offline_masks[self.frame_step], step, return_prob=True)
        curr_losses.append(curr_loss)
        curr_masks.append(curr_mask)
        for _ in range(self.total_offline_frame_num - 3):
            self.update_short_term_memory(
                curr_mask if not use_prev_prob else curr_prob,
                None if use_prev_pred else self.assign_identity(
                    self.offline_one_hot_masks[self.frame_step]))
            self.match_propogate_one_frame()
            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step, return_prob=True)
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)

        aux_loss = torch.cat(aux_losses, dim=0).mean(dim=0)
        pred_loss = torch.cat(curr_losses, dim=0).mean(dim=0)

        loss = aux_weight * aux_loss + pred_loss

        all_pred_mask = aux_masks + curr_masks

        all_frame_loss = aux_losses + curr_losses

        boards = {'image': {}, 'scalar': {}}

        return loss, all_pred_mask, all_frame_loss, boards

    def _init_losses(self):
        cfg = self.cfg

        from networks.layers.loss import CrossEntropyLoss, SoftJaccordLoss
        bce_loss = CrossEntropyLoss(
            cfg.TRAIN_TOP_K_PERCENT_PIXELS,
            cfg.TRAIN_HARD_MINING_RATIO * cfg.TRAIN_TOTAL_STEPS)
        iou_loss = SoftJaccordLoss()

        losses = [bce_loss, iou_loss]
        loss_weights = [0.5, 0.5]

        self.losses = nn.ModuleList(losses)
        self.loss_weights = loss_weights
        self.aux_weight = cfg.TRAIN_AUX_LOSS_WEIGHT
        self.aux_step = cfg.TRAIN_TOTAL_STEPS * cfg.TRAIN_AUX_LOSS_RATIO + 1e-5

    def encode_one_img_mask(self, img=None, mask=None, frame_step=-1):
        if frame_step == -1:
            frame_step = self.frame_step

        if self.enable_offline_enc:
            curr_enc_embs = self.offline_enc_embs[frame_step]
        elif img is None:
            curr_enc_embs = None
        else:
            curr_enc_embs = self.AOT.encode_image(img)

        if mask is not None:
            curr_one_hot_mask = one_hot_mask(mask, self.max_obj_num)
        elif self.enable_offline_enc:
            curr_one_hot_mask = self.offline_one_hot_masks[frame_step]
        else:
            curr_one_hot_mask = None

        return curr_enc_embs, curr_one_hot_mask

    def offline_encoder(self, all_frames, all_masks=None):
        self.enable_offline_enc = True
        self.offline_frames = all_frames.size(0) // self.batch_size

        # extract backbone features
        self.offline_enc_embs = self.split_frames(
            self.AOT.encode_image(all_frames), self.batch_size)
        self.total_offline_frame_num = len(self.offline_enc_embs)

        if all_masks is not None:
            # extract mask embeddings
            offline_one_hot_masks = one_hot_mask(all_masks, self.max_obj_num)
            self.offline_masks = list(
                torch.split(all_masks, self.batch_size, dim=0))
            self.offline_one_hot_masks = list(
                torch.split(offline_one_hot_masks, self.batch_size, dim=0))

        if self.input_size_2d is None:
            self.update_size(all_frames.size()[2:],
                             self.offline_enc_embs[0][-1].size()[2:])

    def assign_identity(self, one_hot_mask):
        if self.enable_id_shuffle:
            one_hot_mask = torch.einsum('bohw,bot->bthw', one_hot_mask,
                                        self.id_shuffle_matrix)

        id_emb = self.AOT.get_id_emb(one_hot_mask).view(
            self.batch_size, -1, self.enc_hw).permute(2, 0, 1)

        if self.training and self.freeze_id:
            id_emb = id_emb.detach()

        return id_emb

    def split_frames(self, xs, chunk_size):
        new_xs = []
        for x in xs:
            all_x = list(torch.split(x, chunk_size, dim=0))
            new_xs.append(all_x)
        return list(zip(*new_xs))

    def add_reference_frame(self,
                            img=None,
                            mask=None,
                            frame_step=-1,
                            obj_nums=None,
                            img_embs=None):
        if self.obj_nums is None and obj_nums is None:
            print('No objects for reference frame!')
            exit()
        elif obj_nums is not None:
            self.obj_nums = obj_nums

        if frame_step == -1:
            frame_step = self.frame_step

        if img_embs is None:
            curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(
                img, mask, frame_step)
        else:
            _, curr_one_hot_mask = self.encode_one_img_mask(
                None, mask, frame_step)
            curr_enc_embs = img_embs

        if curr_enc_embs is None:
            print('No image for reference frame!')
            exit()

        if curr_one_hot_mask is None:
            print('No mask for reference frame!')
            exit()

        if self.input_size_2d is None:
            self.update_size(img.size()[2:], curr_enc_embs[-1].size()[2:])

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        if self.pos_emb is None:
            self.pos_emb = self.AOT.get_pos_emb(curr_enc_embs[-1]).expand(
                self.batch_size, -1, -1,
                -1).view(self.batch_size, -1, self.enc_hw).permute(2, 0, 1)

        curr_id_emb = self.assign_identity(curr_one_hot_mask)
        self.curr_id_embs = curr_id_emb

        # self matching and propagation
        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      None,
                                                      None,
                                                      curr_id_emb,
                                                      pos_emb=self.pos_emb,
                                                      size_2d=self.enc_size_2d)

        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = self.curr_lstt_output

        if self.long_term_memories is None:
            self.init_object_memories(lstt_long_memories, mask)
        else:
            self.update_long_term_memory(lstt_long_memories, mask, is_ref=True)
        self.ref_frame_num += 1
        self.last_mem_step = self.frame_step

        self.short_term_memories_list = [lstt_short_memories]
        self.short_term_memories = lstt_short_memories

    def set_prev_frame(self, img=None, mask=None, frame_step=1):
        self.frame_step = frame_step
        curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(
            img, mask, frame_step)

        if curr_enc_embs is None:
            print('No image for previous frame!')
            exit()

        if curr_one_hot_mask is None:
            print('No mask for previous frame!')
            exit()

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        curr_id_emb = self.assign_identity(curr_one_hot_mask)
        self.curr_id_embs = curr_id_emb

        # self matching and propagation
        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      None,
                                                      None,
                                                      curr_id_emb,
                                                      pos_emb=self.pos_emb,
                                                      size_2d=self.enc_size_2d)

        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = self.curr_lstt_output

        if self.long_term_memories is None:
            self.long_term_memories = lstt_long_memories
        else:
            self.update_long_term_memory(lstt_long_memories, mask)
        self.last_mem_step = frame_step

        self.short_term_memories_list = [lstt_short_memories]
        self.short_term_memories = lstt_short_memories

    def init_object_memories(self, lstt_long_memories, curr_mask):

        self.object_memories = [[] for _ in range((self.max_obj_num + 1))]
        h, w = self.curr_enc_embs[-1].shape[2:]
        downsampled_mask = F.interpolate(curr_mask, (h, w)).view(
                self.batch_size, -1, h*w).permute(2, 0, 1)

        # 1. get patch wised object embedding according to current mask
        for idx in range(self.max_obj_num + 1):
            updated_long_term_memories = []
            for layer, last_long_term_memory in enumerate(lstt_long_memories):
                updated_e = []
                for last_e in last_long_term_memory:
                    if last_e is None:
                        updated_e.append(None)
                    else:
                        obj_e = last_e[(downsampled_mask==idx).squeeze(-1).squeeze(-1)] # patch_num, batch_size, dim
                        updated_e.append(obj_e) 
                updated_long_term_memories.append(updated_e)
            if len(updated_long_term_memories[0][0]):
                self.object_memories[idx].append(updated_long_term_memories)

        # 2. assemble obj mermories to get long term memories
        self.long_term_memories = self.assemble_obj_memories_get_long_term_memories()

    def assemble_obj_memories_get_long_term_memories(self):
        memory_feat_num = len(self.object_memories[0][0][0])        # AOT:[K, V], DeAOT:[K, V, K_id(None), V_id]
        assembled_long_term_memories = []
        for layer in range(len(self.object_memories[0][0])):
            ass_e = [[] for _ in range(memory_feat_num)] 
            total_e = []

            # each object 
            for idx in range(self.max_obj_num + 1):
                # each frame
                frame_e = [[] for _ in range(memory_feat_num)]
                for obj_frame_e in self.object_memories[idx]:
                    # each layer
                    for j, f_e in enumerate(obj_frame_e[layer]):
                        if f_e is None or len(f_e) == 0:
                            continue
                        frame_e[j].append(f_e)
                
                for j in range(len(frame_e)):
                    if j == 2:
                        ass_e[j].append(None)
                    elif len(frame_e[j]) > 0:
                        temp = torch.cat(frame_e[j], dim=0)
                        ass_e[j].append(temp)

            for j in range(len(ass_e)):
                if j == 2:
                    total_e.append(None)
                else:
                    temp = torch.cat(ass_e[j], dim=0)
                    total_e.append(temp)

            assembled_long_term_memories.append(total_e)

        return assembled_long_term_memories

    def cal_obj_patch_num_in_memories(self,obj_idx):
        patch_num = 0

        for frame_e in self.object_memories[obj_idx]:
            for m_e in frame_e:
                for m in m_e:
                    if m is None:
                        continue
                    patch_num += len(m)

        # print(f'obj_idx = {obj_idx}, patch_num = {patch_num}')
        return patch_num   
    
    def cal_obj_memories_sim_dist(self, memory_e, curr_e):
        '''
            Parameters:
                memory_e: obj long term memory embedding, List:[ [ [] * 4 ] * layers_num ] 
                curr_e: obj curr frame embedding
        '''
        total_dist, cnt = 0, 0
        cos_sim = nn.CosineSimilarity()
        for layer_idx, (m_e, c_e) in enumerate(zip(memory_e, curr_e)):
            if layer_idx not in self.cfg.PATCH_SIM_LAYERS_IDX:
                continue
            for memory_idx,(m, c) in enumerate(zip(m_e, c_e)):
                if memory_idx not in self.cfg.PATCH_SIM_MEMORY_IDX:
                    continue
                if c is None or m is None:
                    continue
                # m: (patch_num, batch, dim)
                mean_m = torch.mean(m, dim=0)
                mean_c = torch.mean(c, dim=0)
                total_dist += cos_sim(mean_m, mean_c)
                cnt += 1
        
        return float(total_dist / cnt)

    def drop_memories(self, obj_idx, curr_long_term_memories):
        '''
            Parameters:
                obj_idx
                curr_long_term_memories: curr_embedding 
        '''
        
        frame_max_num = self.cfg.TEST_LONG_TERM_MEM_MAX
        patch_max_num = self.cfg.PATCH_TEST_LONG_TERM_MEM_MAX 

        if not self.cfg.PATCH_WISED_DROP_MEMORIES:
            if len(self.object_memories[obj_idx]) == 0 or len(self.object_memories[obj_idx]) < frame_max_num :
                return
        else:
            if len(self.object_memories[obj_idx]) == 0:
                return
            # cal patch num
            obj_patch_num = self.cal_obj_patch_num_in_memories(obj_idx)
            if obj_patch_num < patch_max_num:
                return

        # only drop the predicted mask
        droped_idx, max_dist = -1, 1
        for frame_idx, frame_e in enumerate(self.object_memories[obj_idx][: -self.ref_frame_num]):
            dist = self.cal_obj_memories_sim_dist(frame_e, curr_long_term_memories)
            if dist < max_dist:
                max_dist = dist
                droped_idx = frame_idx

        # print(f'obj_idx = {obj_idx}, droped frame_idx = {droped_idx}, max_dits = {max_dist}')
        self.object_memories[obj_idx] = self.object_memories[obj_idx][:droped_idx] + self.object_memories[obj_idx][droped_idx + 1:]

    def update_long_term_memory(self, new_long_term_memories, curr_mask, is_ref=False):
        if self.long_term_memories is None:
            self.init_object_memories()

        h, w = self.curr_enc_embs[-1].shape[2:]
        downsampled_mask = F.interpolate(curr_mask, (h, w)).view(
                self.batch_size, -1, h*w).permute(2, 0, 1)

        # 1. update object memories
        for idx in range(self.max_obj_num + 1):
            updated_long_term_memories = []
            for layer, last_long_term_memory in enumerate(new_long_term_memories):
                updated_e = []
                for last_e in last_long_term_memory:
                    if last_e is None:
                        updated_e.append(None)
                    else:
                        obj_e = last_e[(downsampled_mask==idx).squeeze(-1).squeeze(-1)] # patch_num, batch_size, dim
                        updated_e.append(obj_e) 
                updated_long_term_memories.append(updated_e)
            
            if len(updated_long_term_memories[0][0]) == 0:
                continue

            if not self.training:
                self.drop_memories(idx, updated_long_term_memories)

                if is_ref:
                    self.object_memories[idx] += [updated_long_term_memories]
                else:
                    t = [updated_long_term_memories]
                    t += self.object_memories[idx]
                    self.object_memories[idx] = t
            else:
                t = [updated_long_term_memories]
                t += self.object_memories[idx]
                self.object_memories[idx] = t
        # 2. assemble obj mermories to get long term memories
        self.long_term_memories = self.assemble_obj_memories_get_long_term_memories()

    def update_short_term_memory(self, curr_mask, curr_id_emb=None, skip_long_term_update=False):
        if curr_id_emb is None:
            if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                curr_one_hot_mask = one_hot_mask(curr_mask, self.max_obj_num)
            else:
                curr_one_hot_mask = curr_mask
            curr_id_emb = self.assign_identity(curr_one_hot_mask)

        lstt_curr_memories = self.curr_lstt_output[1]
        lstt_curr_memories_2d = []
        for layer_idx in range(len(lstt_curr_memories)):
            curr_k, curr_v = lstt_curr_memories[layer_idx][
                0], lstt_curr_memories[layer_idx][1]
            curr_k, curr_v = self.AOT.LSTT.layers[layer_idx].fuse_key_value_id(
                curr_k, curr_v, curr_id_emb)
            lstt_curr_memories[layer_idx][0], lstt_curr_memories[layer_idx][
                1] = curr_k, curr_v
            lstt_curr_memories_2d.append([
                seq_to_2d(lstt_curr_memories[layer_idx][0], self.enc_size_2d),
                seq_to_2d(lstt_curr_memories[layer_idx][1], self.enc_size_2d)
            ])

        self.short_term_memories_list.append(lstt_curr_memories_2d)
        self.short_term_memories_list = self.short_term_memories_list[
            -self.short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        if self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            # skip the update of long-term memory or not
            if not skip_long_term_update: 
                self.update_long_term_memory(lstt_curr_memories, curr_mask)
            self.last_mem_step = self.frame_step

    def match_propogate_one_frame(self, img=None, img_embs=None):
        self.frame_step += 1
        if img_embs is None:
            curr_enc_embs, _ = self.encode_one_img_mask(
                img, None, self.frame_step)
        else:
            curr_enc_embs = img_embs
        self.curr_enc_embs = curr_enc_embs

        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      self.long_term_memories,
                                                      self.short_term_memories,
                                                      None,
                                                      pos_emb=self.pos_emb,
                                                      size_2d=self.enc_size_2d)

    def decode_current_logits(self, output_size=None):
        curr_enc_embs = self.curr_enc_embs
        curr_lstt_embs = self.curr_lstt_output[0]

        pred_id_logits = self.AOT.decode_id_logits(curr_lstt_embs,
                                                   curr_enc_embs)

        if self.enable_id_shuffle:  # reverse shuffle
            pred_id_logits = torch.einsum('bohw,bto->bthw', pred_id_logits,
                                          self.id_shuffle_matrix)

        # remove unused identities
        for batch_idx, obj_num in enumerate(self.obj_nums):
            pred_id_logits[batch_idx, (obj_num+1):] = - \
                1e+10 if pred_id_logits.dtype == torch.float32 else -1e+4

        self.pred_id_logits = pred_id_logits

        if output_size is not None:
            pred_id_logits = F.interpolate(pred_id_logits,
                                           size=output_size,
                                           mode="bilinear",
                                           align_corners=self.align_corners)

        return pred_id_logits

    def predict_current_mask(self, output_size=None, return_prob=False):
        if output_size is None:
            output_size = self.input_size_2d

        pred_id_logits = F.interpolate(self.pred_id_logits,
                                       size=output_size,
                                       mode="bilinear",
                                       align_corners=self.align_corners)
        pred_mask = torch.argmax(pred_id_logits, dim=1)

        if not return_prob:
            return pred_mask
        else:
            pred_prob = torch.softmax(pred_id_logits, dim=1)
            return pred_mask, pred_prob

    def calculate_current_loss(self, gt_mask, step):
        pred_id_logits = self.pred_id_logits

        pred_id_logits = F.interpolate(pred_id_logits,
                                       size=gt_mask.size()[-2:],
                                       mode="bilinear",
                                       align_corners=self.align_corners)

        label_list = []
        logit_list = []
        for batch_idx, obj_num in enumerate(self.obj_nums):
            now_label = gt_mask[batch_idx].long()
            now_logit = pred_id_logits[batch_idx, :(obj_num + 1)].unsqueeze(0)
            label_list.append(now_label.long())
            logit_list.append(now_logit)

        total_loss = 0
        for loss, loss_weight in zip(self.losses, self.loss_weights):
            total_loss = total_loss + loss_weight * \
                loss(logit_list, label_list, step)

        return total_loss

    def generate_loss_mask(self, gt_mask, step, return_prob=False):
        self.decode_current_logits()
        loss = self.calculate_current_loss(gt_mask, step)
        if return_prob:
            mask, prob = self.predict_current_mask(return_prob=True)
            return loss, mask, prob
        else:
            mask = self.predict_current_mask()
            return loss, mask

    def keep_gt_mask(self, pred_mask, keep_prob=0.2):
        pred_mask = pred_mask.float()
        gt_mask = self.offline_masks[self.frame_step].float().squeeze(1)

        shape = [1 for _ in range(pred_mask.ndim)]
        shape[0] = self.batch_size
        random_tensor = keep_prob + torch.rand(
            shape, dtype=pred_mask.dtype, device=pred_mask.device)
        random_tensor.floor_()  # binarize

        pred_mask = pred_mask * (1 - random_tensor) + gt_mask * random_tensor

        return pred_mask

    def restart_engine(self, batch_size=1, enable_id_shuffle=False):

        self.batch_size = batch_size
        self.frame_step = 0
        self.last_mem_step = -1
        self.enable_id_shuffle = enable_id_shuffle
        self.freeze_id = False

        self.obj_nums = None
        self.pos_emb = None
        self.enc_size_2d = None
        self.enc_hw = None
        self.input_size_2d = None

        self.long_term_memories = None
        self.short_term_memories_list = []
        self.short_term_memories = None

        self.enable_offline_enc = False
        self.offline_enc_embs = None
        self.offline_one_hot_masks = None
        self.offline_frames = -1
        self.total_offline_frame_num = 0
        self.ref_frame_num = 0

        self.curr_enc_embs = None
        self.curr_memories = None
        self.curr_id_embs = None

        if enable_id_shuffle:
            self.id_shuffle_matrix = generate_permute_matrix(
                self.max_obj_num + 1, batch_size, gpu_id=self.gpu_id)
        else:
            self.id_shuffle_matrix = None

    def update_size(self, input_size, enc_size):
        self.input_size_2d = input_size
        self.enc_size_2d = enc_size
        self.enc_hw = self.enc_size_2d[0] * self.enc_size_2d[1]


class DMAOTInferEngine(nn.Module):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 max_aot_obj_num=None,
                 max_len_long_term=9999,):
        super().__init__()

        self.cfg = aot_model.cfg
        self.AOT = aot_model

        if max_aot_obj_num is None or max_aot_obj_num > aot_model.max_obj_num:
            self.max_aot_obj_num = aot_model.max_obj_num
        else:
            self.max_aot_obj_num = max_aot_obj_num

        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.max_len_long_term = max_len_long_term
        self.aot_engines = []

        self.restart_engine()
    def restart_engine(self):
        del (self.aot_engines)
        self.aot_engines = []
        self.obj_nums = None

    def separate_mask(self, mask, obj_nums):
        if mask is None:
            return [None] * len(self.aot_engines)
        if len(self.aot_engines) == 1:
            return [mask], [obj_nums]

        separated_obj_nums = [
            self.max_aot_obj_num for _ in range(len(self.aot_engines))
        ]
        if obj_nums % self.max_aot_obj_num > 0:
            separated_obj_nums[-1] = obj_nums % self.max_aot_obj_num

        if len(mask.size()) == 3 or mask.size()[0] == 1:
            separated_masks = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_mask = ((mask >= start_id) & (mask <= end_id)).float()
                separated_mask = (fg_mask * mask - start_id + 1) * fg_mask
                separated_masks.append(separated_mask)
            return separated_masks, separated_obj_nums
        else:
            prob = mask
            separated_probs = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_prob = prob[start_id:(end_id + 1)]
                bg_prob = 1. - torch.sum(fg_prob, dim=1, keepdim=True)
                separated_probs.append(torch.cat([bg_prob, fg_prob], dim=1))
            return separated_probs, separated_obj_nums

    def min_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_logits = []
        bg_logits = []

        for logit in all_logits:
            bg_logits.append(logit[:, 0:1])
            fg_logits.append(logit[:, 1:1 + self.max_aot_obj_num])

        bg_logit, _ = torch.min(torch.cat(bg_logits, dim=1),
                                dim=1,
                                keepdim=True)
        merged_logit = torch.cat([bg_logit] + fg_logits, dim=1)

        return merged_logit

    def soft_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_probs = []
        bg_probs = []

        for logit in all_logits:
            prob = torch.softmax(logit, dim=1)
            bg_probs.append(prob[:, 0:1])
            fg_probs.append(prob[:, 1:1 + self.max_aot_obj_num])

        bg_prob = torch.prod(torch.cat(bg_probs, dim=1), dim=1, keepdim=True)
        merged_prob = torch.cat([bg_prob] + fg_probs,
                                dim=1).clamp(1e-5, 1 - 1e-5)
        merged_logit = torch.logit(merged_prob)

        return merged_logit

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = DMAOTEngine(self.AOT, self.gpu_id,
                                   self.long_term_mem_gap,
                                   self.short_term_mem_skip,
                                   self.max_len_long_term,)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(
            mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(
                self.aot_engines, separated_masks, separated_obj_nums):
            aot_engine.add_reference_frame(img,
                                        separated_mask,
                                        obj_nums=[separated_obj_num],
                                        frame_step=frame_step,
                                        img_embs=img_embs)
                
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()

    def match_propogate_one_frame(self, img=None):
        img_embs = None
        for aot_engine in self.aot_engines:
            aot_engine.match_propogate_one_frame(img, img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

    def decode_current_logits(self, output_size=None):
        all_logits = []
        for aot_engine in self.aot_engines:
            all_logits.append(aot_engine.decode_current_logits(output_size))
        pred_id_logits = self.soft_logit_aggregation(all_logits)
        return pred_id_logits

    def update_memory(self, curr_mask, skip_long_term_update=False):
        _curr_mask = F.interpolate(curr_mask,self.input_size_2d)
        separated_masks, _ = self.separate_mask(_curr_mask, self.obj_nums)
        for aot_engine, separated_mask in zip(self.aot_engines,
                                              separated_masks):
            aot_engine.update_short_term_memory(separated_mask, 
                                                skip_long_term_update=skip_long_term_update)

    def update_size(self):
        self.input_size_2d = self.aot_engines[0].input_size_2d
        self.enc_size_2d = self.aot_engines[0].enc_size_2d
        self.enc_hw = self.aot_engines[0].enc_hw

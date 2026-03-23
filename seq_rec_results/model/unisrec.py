import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer"""
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor with optional image/review expert streams.

    Original behaviour (n_img_exps=0):
        gate [*, n_exps] from text_emb -> weight text experts -> [*, out_dim]

    With extra streams enabled:
        gate [*, n_exps + n_img_exps + n_review_exps] from text_emb
        text experts (n_exps):     text_emb -> [*, out_dim]
        image experts (n_img_exps): x_img   -> [*, out_dim]
        review experts (n_review_exps): x_review -> [*, out_dim]
        all experts weighted by shared gate -> sum -> [*, out_dim]

    When an extra stream is None at runtime its gate slice is masked out and
    the remaining active experts are renormalized (safe fallback).
    """

    def __init__(self, n_exps, layers, dropout=0.0, noise=True,
                 n_img_exps=0, img_in_dim=0, n_review_exps=0, review_in_dim=0):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.n_img_exps = n_img_exps
        self.n_review_exps = n_review_exps
        self.noisy_gating = noise
        n_total = n_exps + n_img_exps + n_review_exps

        in_dim, out_dim = layers[0], layers[1]

        # Text experts (original)
        self.experts = nn.ModuleList(
            [PWLayer(in_dim, out_dim, dropout) for _ in range(n_exps)]
        )

        # Image experts (only when n_img_exps > 0)
        self.img_experts = nn.ModuleList(
            [PWLayer(img_in_dim, out_dim, dropout) for _ in range(n_img_exps)]
        ) if n_img_exps > 0 else nn.ModuleList()
        self.review_experts = nn.ModuleList(
            [PWLayer(review_in_dim, out_dim, dropout) for _ in range(n_review_exps)]
        ) if n_review_exps > 0 else nn.ModuleList()

        # Shared gate driven by text_emb; size = n_total
        self.w_gate  = nn.Parameter(torch.zeros(in_dim, n_total), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(in_dim, n_total), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train: # 如果启用noisy-gating，加噪声
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits
        return F.softmax(logits, dim=-1)   # [*, n_total]，每个expert权重

    def forward(self, x, x_img=None, x_review=None):
        """
        Args:
            x:     text embedding  [*, in_dim]
            x_img: image embedding [*, img_in_dim] or None
            x_review: review embedding [*, review_in_dim] or None
        Returns:
            [*, out_dim]
        """
        gates = self.noisy_top_k_gating(x, self.training)  # [*, n_total]

        # Text expert outputs
        text_out = [e(x).unsqueeze(-2) for e in self.experts]   # each [*, 1, out_dim]

        active_out = list(text_out)
        gate_parts = [gates[..., :self.n_exps]]
        offset = self.n_exps

        if self.n_img_exps > 0 and x_img is not None:
            img_out = [e(x_img).unsqueeze(-2) for e in self.img_experts]
            active_out.extend(img_out)
            gate_parts.append(gates[..., offset: offset + self.n_img_exps])
        offset += self.n_img_exps

        if self.n_review_exps > 0 and x_review is not None:
            review_out = [e(x_review).unsqueeze(-2) for e in self.review_experts]
            active_out.extend(review_out)
            gate_parts.append(gates[..., offset: offset + self.n_review_exps])

        all_out = torch.cat(active_out, dim=-2)
        gates = torch.cat(gate_parts, dim=-1)
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        return (gates.unsqueeze(-1) * all_out).sum(dim=-2)      # [*, out_dim]


class UniSRec(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        def _cfg(key, default=None):
            try:
                return config[key]
            except Exception:
                return default

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']

        self.ft_loss = _cfg('ft_loss', 'ce')
        self.inbatch_mask_same_item = _cfg('inbatch_mask_same_item', True)

        self.freeze_item_embedding = _cfg('freeze_item_embedding', False)
        self.freeze_item_embedding_init = _cfg('freeze_item_embedding_init', 'keep')

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None # 不含ID embedding
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        if self.train_stage == 'transductive_ft' and self.freeze_item_embedding:
            if getattr(self, 'item_embedding', None) is not None:
                if str(self.freeze_item_embedding_init).lower() == 'zero':
                    nn.init.zeros_(self.item_embedding.weight)
                self.item_embedding.weight.requires_grad = False

        # ------------------------------------------------------------------ #
        # Optional image embedding stream
        # Set img_plm_suffix + img_plm_size in config; file must exist.
        # When absent, model degrades to original single-stream MoE.
        # ------------------------------------------------------------------ #
        self.img_embedding = None
        self.img_plm_size = int(_cfg('img_plm_size', 0) or 0)
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            if getattr(dataset, 'img_embedding', None) is not None:
                self.img_embedding = copy.deepcopy(dataset.img_embedding)

        self.review_embedding = None
        self.review_plm_size = int(_cfg('review_plm_size', 0) or 0)
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            if getattr(dataset, 'review_embedding', None) is not None:
                self.review_embedding = copy.deepcopy(dataset.review_embedding)
            # else:
            #     self.review_embedding = self._load_feature_embedding(
            #         config, dataset, _cfg('review_plm_suffix', ''), self.review_plm_size
            #     )

        n_img_exps = int(_cfg('n_img_exps', 4)) if self.img_embedding is not None else 0
        img_in_dim = self.img_plm_size if self.img_embedding is not None else 0
        n_review_exps = int(_cfg('n_review_exps', 4)) if self.review_embedding is not None else 0

        review_in_dim = self.review_plm_size if self.review_embedding is not None else 0

        self.moe_adaptor = MoEAdaptorLayer(
            n_exps=config['n_exps'],
            layers=config['adaptor_layers'],
            dropout=config['adaptor_dropout_prob'],
            n_img_exps=n_img_exps,
            img_in_dim=img_in_dim,
            n_review_exps=n_review_exps,
            review_in_dim=review_in_dim,
        )

        # Cache for full-item fused vectors; invalidated when switching to train().
        self._item_emb_cache = None

    # ---------------------------------------------------------------------- #
    # Cache management                                                        #
    # ---------------------------------------------------------------------- #

    def train(self, mode: bool = True):
        if mode:
            self._item_emb_cache = None
        return super().train(mode)

    @torch.no_grad()
    def _build_item_emb_cache(self):
        """Compute normalised full-item vectors once per eval epoch."""
        plm_w = self.plm_embedding.weight                          # [n, 768], frozen
        img_w = self.img_embedding.weight if self.img_embedding is not None else None
        review_w = self.review_embedding.weight if self.review_embedding is not None else None
        item_emb = self.moe_adaptor(plm_w, img_w, review_w)       # [n, 300]
        if self.train_stage == 'transductive_ft' and self.item_embedding is not None:
            item_emb = item_emb + self.item_embedding.weight
        return F.normalize(item_emb, dim=-1)

    def _get_item_emb_cache(self):
        if self._item_emb_cache is None:
            self._item_emb_cache = self._build_item_emb_cache()
        return self._item_emb_cache

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    def _img_seq(self, item_seq):
        """Image embeddings for a sequence tensor; None if no img_embedding."""
        if self.img_embedding is None:
            return None
        return self.img_embedding(item_seq)   # [B, L, img_plm_size]

    def _img_items(self, item_ids):
        """Image embeddings for a 1-D item id tensor; None if no img_embedding."""
        if self.img_embedding is None:
            return None
        return self.img_embedding(item_ids)   # [B, img_plm_size]

    def _review_seq(self, item_seq):
        if self.review_embedding is None:
            return None
        return self.review_embedding(item_seq)

    def _review_items(self, item_ids):
        if self.review_embedding is None:
            return None
        return self.review_embedding(item_ids)

    def _load_feature_embedding(self, config, dataset, suffix, dim):
        if not suffix or int(dim or 0) <= 0:
            return None

        dataset_name = str(config['dataset'])
        data_path = str(config['data_path'])
        file_name = suffix if str(suffix).startswith(f'{dataset_name}.') else f'{dataset_name}.{suffix}'
        feat_path = os.path.join(data_path, dataset_name, file_name)
        if not os.path.isabs(feat_path):
            feat_path = os.path.abspath(feat_path)
        if not os.path.exists(feat_path):
            return None

        rows = int(getattr(self, 'n_items', 0) or getattr(dataset, 'item_num', 0) or 0) - 1
        if rows <= 0 and getattr(self, 'plm_embedding', None) is not None:
            rows = int(self.plm_embedding.weight.size(0)) - 1
        if rows <= 0:
            return None

        expected = rows * int(dim) * 4
        actual = os.path.getsize(feat_path)
        if actual != expected:
            raise ValueError(
                f'Unexpected feature file size for {feat_path}: {actual}, expected {expected}'
            )

        feat = np.memmap(feat_path, dtype=np.float32, mode='r', shape=(rows, int(dim)))
        pad = np.zeros((1, int(dim)), dtype=np.float32)
        weight = np.concatenate([pad, np.asarray(feat)], axis=0)
        emb = nn.Embedding.from_pretrained(torch.from_numpy(weight), freeze=True, padding_idx=0)
        return emb

    # ---------------------------------------------------------------------- #
    # Forward (unchanged from original)                                       #
    # ---------------------------------------------------------------------- #

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.train_stage == 'transductive_ft' and self.item_embedding is not None:
            input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B, H]

    # ---------------------------------------------------------------------- #
    # Pre-train tasks (unchanged)                                             #
    # ---------------------------------------------------------------------- #

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_items_emb = self.moe_adaptor(interaction['pos_item_emb'])
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        # 通过same_pos_id屏蔽正样本
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def seq_seq_contrastive_task(self, seq_output, same_pos_id, interaction):
        # 增强序列，经过item drop和word drop
        item_seq_aug = interaction[self.ITEM_SEQ + '_aug']
        item_seq_len_aug = interaction[self.ITEM_SEQ_LEN + '_aug']
        item_emb_list_aug = self.moe_adaptor(interaction['item_emb_list_aug'])
        seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug) # 获得增强序列表示
        seq_output_aug = F.normalize(seq_output_aug, dim=1)

        # 正样本：原序列vs自己的增强序列
        # 负样本：原序列vsbatch中其他序列的增强序列
        pos_logits = (seq_output * seq_output_aug).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(interaction['item_emb_list'])
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        pos_id = interaction['item_id']
        # 如果batch内两个样本的正样本相同，那么它们不能互当负样本
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        loss_seq_item = self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
        loss_seq_seq = self.seq_seq_contrastive_task(seq_output, same_pos_id, interaction)
        return loss_seq_item + self.lam * loss_seq_seq # 最终损失是seq-item和seq-seq对比损失的加权
    
    # ---------------------------------------------------------------------- #
    # Fine-tune loss（给RecBole推荐任务时调用）                                                          #
    # ---------------------------------------------------------------------- #

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # Sequence side: moe with optional img stream
        item_emb_list = self.moe_adaptor(
            self.plm_embedding(item_seq),   # [B, L, 768]
            self._img_seq(item_seq),        # [B, L, img_dim] or None
            self._review_seq(item_seq),     # [B, L, review_dim] or None
        )
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        pos_items = interaction[self.POS_ITEM_ID]

        if self.ft_loss == 'inbatch':
            pos_items_emb = self.moe_adaptor(
                self.plm_embedding(pos_items),   # [B, 768]
                self._img_items(pos_items),      # [B, img_dim] or None
                self._review_items(pos_items),   # [B, review_dim] or None
            )
            if self.train_stage == 'transductive_ft' and self.item_embedding is not None:
                pos_items_emb = pos_items_emb + self.item_embedding(pos_items)
            pos_items_emb = F.normalize(pos_items_emb, dim=1)

            logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
            if self.inbatch_mask_same_item:
                same = (pos_items.unsqueeze(1) == pos_items.unsqueeze(0))
                eye  = torch.eye(pos_items.shape[0], dtype=torch.bool, device=pos_items.device)
                same = torch.logical_and(same, ~eye)
                logits = torch.where(same, torch.full_like(logits, -1e9), logits)
            labels = torch.arange(pos_items.shape[0], device=pos_items.device)
            return F.cross_entropy(logits, labels)

        # Full-softmax CE: all item vectors under no_grad.
        # plm_embedding / img_embedding / review_embedding are frozen.
        # moe_adaptor params get their gradients from the sequence-side call above.
        with torch.no_grad():
            test_item_emb = self.moe_adaptor(
                self.plm_embedding.weight,       # [n, 768]
                self.img_embedding.weight if self.img_embedding is not None else None,
                self.review_embedding.weight if self.review_embedding is not None else None,
            )
            if self.train_stage == 'transductive_ft' and self.item_embedding is not None:
                test_item_emb = test_item_emb + self.item_embedding.weight
            test_item_emb = F.normalize(test_item_emb, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        return self.loss_fct(logits, pos_items)

    # ---------------------------------------------------------------------- #
    # Predict                                                                  #
    # ---------------------------------------------------------------------- #

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        item_emb_list = self.moe_adaptor(
            self.plm_embedding(item_seq),
            self._img_seq(item_seq),
            self._review_seq(item_seq),
        )
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=-1)

        # Cached full-item vectors (built once per eval epoch, no_grad)
        test_items_emb = self._get_item_emb_cache()   # [n, 300]

        return torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n]

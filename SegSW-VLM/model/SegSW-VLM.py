from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN,IMAGE_TOKEN_INDEX)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
import cv2
def masks_noise(masks):
	def get_incoherent_mask(input_masks, sfact):
		mask = input_masks.float()
		w = input_masks.shape[-1]
		h = input_masks.shape[-2]
		mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
		mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
		mask_residue = (mask - mask_recover).abs()
		mask_residue = (mask_residue >= 0.01).float()
		return mask_residue
	gt_masks_vector = masks / 255
	mask_noise = torch.randn(gt_masks_vector.shape, device= gt_masks_vector.device) * 1.0
	inc_masks = get_incoherent_mask(gt_masks_vector,  8)
	gt_masks_vector = ((gt_masks_vector + mask_noise * inc_masks) > 0.5).float()
	gt_masks_vector = gt_masks_vector * 255

	return gt_masks_vector

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def soft_iou_loss(inputs, targets, num_masks, eps=1e-6):
    probabilities = inputs.sigmoid().flatten(1, 2)
    targets = targets.flatten(1, 2)
    intersection = (probabilities * targets).sum(dim=-1)
    union = probabilities.sum(dim=-1) + targets.sum(dim=-1) - intersection
    loss = 1.0 - (intersection + eps) / (union + eps)
    return loss.sum() / (num_masks + 1e-8)


def boundary_dice_loss(inputs, targets, num_masks, eps=1e-6):
    probabilities = inputs.sigmoid().unsqueeze(1)
    targets = targets.unsqueeze(1)
    pred_boundary = torch.abs(
        F.max_pool2d(probabilities, 3, stride=1, padding=1)
        - (-F.max_pool2d(-probabilities, 3, stride=1, padding=1))
    )
    target_boundary = torch.abs(
        F.max_pool2d(targets, 3, stride=1, padding=1)
        - (-F.max_pool2d(-targets, 3, stride=1, padding=1))
    )
    pred_boundary = pred_boundary.flatten(1)
    target_boundary = target_boundary.flatten(1)
    numerator = 2.0 * (pred_boundary * target_boundary).sum(dim=-1)
    denominator = pred_boundary.sum(dim=-1) + target_boundary.sum(dim=-1)
    loss = 1.0 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + 1e-8)


class WasteCandidateMining(nn.Module):
    """WCM: mine top-K dense visual tokens as candidate waste evidence."""

    def __init__(self, dim=256, topk=64):
        super().__init__()
        self.topk = int(topk)
        self.score = nn.Conv2d(dim, 1, kernel_size=1)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, image_embeddings):
        b, c, h, w = image_embeddings.shape
        feat = image_embeddings.flatten(2).transpose(1, 2)  # B, HW, C
        scores = self.score(image_embeddings).flatten(2).squeeze(1)  # B, HW
        k = min(max(self.topk, 1), scores.shape[-1])
        _, idx = torch.topk(scores, k=k, dim=-1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, c)
        candidates = torch.gather(feat, dim=1, index=gather_idx)
        candidates = self.token_proj(candidates)
        global_context = feat.mean(dim=1, keepdim=True).expand(-1, k, -1)
        candidates = candidates + self.context_proj(torch.cat([candidates, global_context], dim=-1))
        return candidates


class StructuredAttributeParser(nn.Module):
    """SAP: convert structured waste attribute ids into a semantic prior."""

    def __init__(self, dim=256, vocab_sizes=(8, 20, 8, 24, 8, 8)):
        super().__init__()
        self.num_attrs = len(vocab_sizes)
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim) for size in vocab_sizes])
        self.reliability = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, attribute_ids, context=None):
        if attribute_ids is None:
            return None
        if attribute_ids.dim() == 1:
            attribute_ids = attribute_ids.unsqueeze(0)
        attribute_ids = attribute_ids.long()
        attr_embs = []
        for i, emb in enumerate(self.embeddings):
            ids = attribute_ids[:, i].clamp(min=0, max=emb.num_embeddings - 1)
            attr_embs.append(emb(ids))
        attr_embs = torch.stack(attr_embs, dim=1)  # B, A, C
        reliability_input = attr_embs
        if context is not None:
            context = context.to(dtype=attr_embs.dtype, device=attr_embs.device)
            reliability_input = attr_embs + context.unsqueeze(1)
        weights = torch.softmax(self.reliability(reliability_input).squeeze(-1), dim=-1)
        prior = (attr_embs * weights.unsqueeze(-1)).sum(dim=1)
        return self.fusion(prior), weights


class SolidWasteQueryFormer(nn.Module):
    """SWQ-Former: learn waste queries from all multimodal hidden states."""

    def __init__(self, input_dim, dim=256, num_queries=4, num_heads=8, num_layers=2):
        super().__init__()
        self.num_queries = int(num_queries)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
        )
        self.queries = nn.Parameter(torch.empty(1, self.num_queries, dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_q": nn.LayerNorm(dim),
                        "cross_attn": nn.MultiheadAttention(
                            dim, num_heads, batch_first=True
                        ),
                        "norm_ffn": nn.LayerNorm(dim),
                        "ffn": nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                    }
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, hidden_states, key_padding_mask=None):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if hidden_states.dim() != 3:
            raise ValueError(
                "SWQ-Former expects [B, L, C], got {}".format(
                    tuple(hidden_states.shape)
                )
            )

        memory = self.input_proj(hidden_states)

        if key_padding_mask is not None:
            if key_padding_mask.dim() == 1:
                key_padding_mask = key_padding_mask.unsqueeze(0)

            key_padding_mask = key_padding_mask.reshape(
                key_padding_mask.shape[0], -1
            )

            if key_padding_mask.shape[0] != memory.shape[0]:
                key_padding_mask = None
            elif key_padding_mask.shape[1] > memory.shape[1]:
                key_padding_mask = key_padding_mask[:, :memory.shape[1]]
            elif key_padding_mask.shape[1] < memory.shape[1]:
                pad = torch.zeros(
                    key_padding_mask.shape[0],
                    memory.shape[1] - key_padding_mask.shape[1],
                    dtype=torch.bool,
                    device=key_padding_mask.device,
                )
                key_padding_mask = torch.cat(
                    [key_padding_mask.bool(), pad], dim=1
                )

            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.bool()

        queries = self.queries.expand(memory.shape[0], -1, -1).to(
            dtype=memory.dtype, device=memory.device
        )
        for layer in self.layers:
            q = layer["norm_q"](queries)
            delta, _ = layer["cross_attn"](
                q,
                memory,
                memory,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            queries = queries + delta
            queries = queries + layer["ffn"](layer["norm_ffn"](queries))
        return queries


class HierarchicalEvidenceConsensusGrounding(nn.Module):
    """HECG: build an explicit multi-scale evidence field and ground SWQs."""

    def __init__(self, dim=256, topk=64, scales=(1, 2, 4)):
        super().__init__()
        self.topk = int(topk)
        self.scales = tuple(int(scale) for scale in scales)
        self.query_proto = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.visual_proj = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=1) for _ in self.scales]
        )
        self.visual_score = nn.ModuleList(
            [nn.Conv2d(dim, 1, kernel_size=1) for _ in self.scales]
        )
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, len(self.scales)),
        )
        self.ground_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.output_norm = nn.LayerNorm(dim)

    def _candidate_evidence(self, feature, prototype, visual_proj, score_head):
        feature = visual_proj(feature)
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        tokens_norm = F.normalize(tokens.float(), dim=-1).to(dtype=tokens.dtype)
        proto_norm = F.normalize(prototype.float(), dim=-1).to(dtype=prototype.dtype)
        semantic_score = torch.einsum("bmc,bnc->bmn", proto_norm, tokens_norm).mean(dim=1)
        visual_score = score_head(feature).flatten(2).squeeze(1)
        score = semantic_score + visual_score

        k = min(max(self.topk, 1), score.shape[-1])
        top_values, top_indices = torch.topk(score, k=k, dim=-1)
        sparse = score.new_full(score.shape, -20.0)
        sparse.scatter_(1, top_indices, top_values)
        evidence = torch.sigmoid(sparse.view(b, 1, h, w))

        # Local consensus connects fragmented neighboring evidence and suppresses
        # isolated candidates without introducing a text-to-vision adapter.
        neighbor = F.avg_pool2d(evidence, kernel_size=3, stride=1, padding=1)
        evidence = torch.clamp(evidence + neighbor, min=0.0, max=1.0)
        return feature, evidence

    def forward(self, queries, image_embedding):
        prototype = self.query_proto(queries)
        base_h, base_w = image_embedding.shape[-2:]
        evidence_maps = []
        scale_features = []
        for index, scale in enumerate(self.scales):
            if scale == 1:
                feature = image_embedding
            else:
                target_h = max(1, base_h // scale)
                target_w = max(1, base_w // scale)
                feature = F.adaptive_avg_pool2d(image_embedding, (target_h, target_w))
            feature, evidence = self._candidate_evidence(
                feature,
                prototype,
                self.visual_proj[index],
                self.visual_score[index],
            )
            scale_features.append(feature)
            evidence_maps.append(
                F.interpolate(
                    evidence,
                    size=(base_h, base_w),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        global_visual = image_embedding.flatten(2).mean(dim=-1)
        global_query = queries.mean(dim=1)
        scale_weights = torch.softmax(
            self.scale_gate(torch.cat([global_query, global_visual], dim=-1)),
            dim=-1,
        )
        evidence_stack = torch.stack(evidence_maps, dim=1)
        evidence_field = (
            evidence_stack * scale_weights[:, :, None, None, None]
        ).sum(dim=1)

        flat_feature = image_embedding.flatten(2).transpose(1, 2)
        flat_evidence = evidence_field.flatten(2).transpose(1, 2)
        normalized_evidence = flat_evidence / (
            flat_evidence.sum(dim=1, keepdim=True) + 1e-6
        )
        visual_evidence = (flat_feature * normalized_evidence).sum(dim=1)
        visual_evidence = visual_evidence.unsqueeze(1).expand(-1, queries.shape[1], -1)
        grounded_delta = self.ground_proj(
            torch.cat([queries, visual_evidence], dim=-1)
        )
        grounded_queries = self.output_norm(queries + grounded_delta)

        upsampled = evidence_stack.squeeze(2)
        consensus_target = evidence_field.detach()
        consensus_loss = sum(
            F.l1_loss(upsampled[:, i : i + 1], consensus_target)
            for i in range(upsampled.shape[1])
        ) / max(upsampled.shape[1], 1)
        return grounded_queries, evidence_field, consensus_loss, scale_weights


class ContrastiveAmbiguityPromptGenerator(nn.Module):
    """CAPG: generate target, boundary, and distractor prompt roles."""

    def __init__(self, dim=256, margin=0.2):
        super().__init__()
        self.margin = float(margin)
        self.hardness = nn.Conv2d(dim, 1, kernel_size=1)
        self.target_head = self._make_head(dim)
        self.boundary_head = self._make_head(dim)
        self.distractor_head = self._make_head(dim)
        self.role_embeddings = nn.Parameter(torch.empty(1, 3, dim))
        nn.init.normal_(self.role_embeddings, mean=0.0, std=0.02)

    @staticmethod
    def _make_head(dim):
        return nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def _weighted_pool(feature, weight):
        weighted = feature * weight
        return weighted.flatten(2).sum(dim=-1) / (
            weight.flatten(2).sum(dim=-1) + 1e-6
        )

    def forward(self, grounded_queries, image_embedding, evidence_field):
        query = grounded_queries.mean(dim=1)
        ambiguity = 4.0 * evidence_field * (1.0 - evidence_field)
        hardness = torch.sigmoid(self.hardness(image_embedding))
        target_visual = self._weighted_pool(image_embedding, evidence_field)
        boundary_visual = self._weighted_pool(image_embedding, ambiguity)
        distractor_weight = (1.0 - evidence_field) * hardness
        distractor_visual = self._weighted_pool(image_embedding, distractor_weight)

        target_prompt = self.target_head(torch.cat([query, target_visual], dim=-1))
        boundary_prompt = self.boundary_head(
            torch.cat([query, boundary_visual], dim=-1)
        )
        distractor_prompt = self.distractor_head(
            torch.cat([query, distractor_visual], dim=-1)
        )
        raw_prompts = torch.stack(
            [target_prompt, boundary_prompt, distractor_prompt], dim=1
        )
        polarity = raw_prompts.new_tensor([1.0, 1.0, -1.0]).view(1, 3, 1)
        prompts = (raw_prompts + self.role_embeddings.to(raw_prompts.dtype)) * polarity

        target_similarity = F.cosine_similarity(target_prompt, target_visual, dim=-1)
        negative_similarity = F.cosine_similarity(
            target_prompt, distractor_visual, dim=-1
        )
        contrast_loss = F.relu(
            self.margin - target_similarity + negative_similarity
        ).mean()
        diversity_loss = (
            F.cosine_similarity(target_prompt, distractor_prompt, dim=-1).abs().mean()
            + 0.5
            * F.cosine_similarity(boundary_prompt, distractor_prompt, dim=-1)
            .abs()
            .mean()
        )
        return prompts, ambiguity, contrast_loss, diversity_loss


class CandidateGroundedTextAlignment(nn.Module):
    """CGTA: align SWQ with candidate visual evidence instead of whole-image features."""

    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        self.query_fuse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.calibrate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, swq, candidate_tokens, attr_prior=None):
        q = swq
        if attr_prior is not None:
            q = q + attr_prior.unsqueeze(1).to(dtype=q.dtype, device=q.device)
        q = self.query_fuse(q)
        evidence, _ = self.cross_attn(q, candidate_tokens, candidate_tokens)
        gate = self.gate(torch.cat([q, evidence], dim=-1))
        delta = self.calibrate(torch.cat([q, evidence, q * evidence], dim=-1))
        return q + gate * delta


class DualPromptDecoder(nn.Module):
    """DPD: generate positive waste and negative background prompt tokens."""

    def __init__(self, dim=256):
        super().__init__()
        self.positive_prompt = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.negative_prompt = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, aligned_query):
        pos = self.positive_prompt(aligned_query)
        neg = self.negative_prompt(aligned_query)
        return torch.cat([pos, neg], dim=1)


class SegSWVLMMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(SegSWVLMMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.config.text_gate_init = kwargs.get("text_gate_init", -10.0)
            self.config.train_sam_encoder = kwargs.get("train_sam_encoder", True)
            self.config.sam_encoder_trainable_layers = kwargs.get("sam_encoder_trainable_layers", 6)
            self.config.use_wcm = kwargs.get("use_wcm", False)
            self.config.use_sap = kwargs.get("use_sap", False)
            self.config.use_cgta = kwargs.get("use_cgta", False)
            self.config.use_dpd = kwargs.get("use_dpd", False)
            self.config.wcm_topk = kwargs.get("wcm_topk", 64)
            self.config.use_swq_former = kwargs.get("use_swq_former", False)
            self.config.use_hecg = kwargs.get("use_hecg", False)
            self.config.use_capg = kwargs.get("use_capg", False)
            self.config.swq_num_queries = kwargs.get("swq_num_queries", 4)
            self.config.swq_num_layers = kwargs.get("swq_num_layers", 2)
            self.config.swq_num_heads = kwargs.get("swq_num_heads", 8)
            self.config.hecg_scales = kwargs.get("hecg_scales", (1, 2, 4))
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_segswvlm_modules(self.config)

    def initialize_segswvlm_modules(self, config):
        # SAM即视觉编码加mask生成，前者冻结后者可训练
        self.visual_model = build_sam_vit_h(self.vision_pretrained)#初始化SAM预训练参数
        for param in self.visual_model.parameters():
            param.requires_grad = False#冻结视觉编码部分参数
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()#mask解码部分训练参数可更新
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer，MLP初始化可训练
        if getattr(config, "train_sam_encoder", True):
            self.visual_model.image_encoder.train()
            trainable_layers = int(getattr(config, "sam_encoder_trainable_layers", 6))
            if hasattr(self.visual_model.image_encoder, "neck"):
                for param in self.visual_model.image_encoder.neck.parameters():
                    param.requires_grad = True
            if hasattr(self.visual_model.image_encoder, "blocks"):
                blocks = self.visual_model.image_encoder.blocks
                selected_blocks = blocks if trainable_layers <= 0 else blocks[-trainable_layers:]
                for block in selected_blocks:
                    for param in block.parameters():
                        param.requires_grad = True

        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        text_gate_init = float(getattr(config, "text_gate_init", -10.0))
        self.text_gate_logit = nn.Parameter(torch.tensor(text_gate_init))
        self.text_gate_logit.requires_grad = True
        self.visual_seg_prompt = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.normal_(self.visual_seg_prompt, mean=0.0, std=0.02)
        self.visual_seg_prompt.requires_grad = True

        out_dim = config.out_dim
        if getattr(config, "use_wcm", False):
            self.wcm = WasteCandidateMining(dim=out_dim, topk=getattr(config, "wcm_topk", 64))
            for param in self.wcm.parameters():
                param.requires_grad = True
        if getattr(config, "use_sap", False):
            self.sap = StructuredAttributeParser(dim=out_dim)
            for param in self.sap.parameters():
                param.requires_grad = True
        if getattr(config, "use_cgta", False):
            self.cgta = CandidateGroundedTextAlignment(dim=out_dim)
            for param in self.cgta.parameters():
                param.requires_grad = True
        if getattr(config, "use_dpd", False):
            self.dpd = DualPromptDecoder(dim=out_dim)
            for param in self.dpd.parameters():
                param.requires_grad = True
        if getattr(config, "use_swq_former", False):
            self.swq_former = SolidWasteQueryFormer(
                input_dim=config.hidden_size,
                dim=out_dim,
                num_queries=getattr(config, "swq_num_queries", 4),
                num_heads=getattr(config, "swq_num_heads", 8),
                num_layers=getattr(config, "swq_num_layers", 2),
            )
            for param in self.swq_former.parameters():
                param.requires_grad = True
        if getattr(config, "use_hecg", False):
            self.hecg = HierarchicalEvidenceConsensusGrounding(
                dim=out_dim,
                topk=getattr(config, "wcm_topk", 64),
                scales=getattr(config, "hecg_scales", (1, 2, 4)),
            )
            for param in self.hecg.parameters():
                param.requires_grad = True
        if getattr(config, "use_capg", False):
            self.capg = ContrastiveAmbiguityPromptGenerator(dim=out_dim)
            for param in self.capg.parameters():
                param.requires_grad = True

    def get_text_gate(self):
        return torch.sigmoid(self.text_gate_logit)


class SegSWVLMModel(SegSWVLMMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(SegSWVLMModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class SegSWVLMForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            
        else:
            config.mm_vision_tower = config.vision_tower
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.mask_loss_type = kwargs.pop("mask_loss_type", "focal_bce_dice")
        self.focal_alpha = kwargs.pop("focal_alpha", 0.25)
        self.focal_gamma = kwargs.pop("focal_gamma", 2.0)
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.text_gate_init = kwargs.get("text_gate_init", -10.0)
        self.use_wcm = kwargs.get("use_wcm", False)
        self.use_sap = kwargs.get("use_sap", False)
        self.use_cgta = kwargs.get("use_cgta", False)
        self.use_dpd = kwargs.get("use_dpd", False)
        self.use_swq_former = kwargs.get("use_swq_former", False)
        self.use_hecg = kwargs.get("use_hecg", False)
        self.use_capg = kwargs.get("use_capg", False)
        self.evidence_consensus_loss_weight = kwargs.pop(
            "evidence_consensus_loss_weight", 0.1
        )
        self.prompt_contrast_loss_weight = kwargs.pop(
            "prompt_contrast_loss_weight", 0.1
        )
        self.prompt_diversity_loss_weight = kwargs.pop(
            "prompt_diversity_loss_weight", 0.05
        )
        self.evidence_supervision_loss_weight = kwargs.pop(
            "evidence_supervision_loss_weight", 0.1
        )
        self.boundary_loss_weight = kwargs.pop("boundary_loss_weight", 0.0)
        self.soft_iou_loss_weight = kwargs.pop("soft_iou_loss_weight", 0.0)
        super().__init__(config)

        self.model = SegSWVLMModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        train_image_encoder = self.training and any(
            param.requires_grad for param in self.model.visual_model.image_encoder.parameters()
        )
        grad_context = torch.enable_grad() if train_image_encoder else torch.no_grad()
        with grad_context:
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def get_text_gate(self):
        return self.model.get_text_gate()

    def apply_text_gate(self, text_embeding):
        gate = self.get_text_gate().to(dtype=text_embeding.dtype, device=text_embeding.device)
        return text_embeding * gate

    def build_seg_prompt(self, text_embeding):
        visual_prompt = self.model.visual_seg_prompt.to(
            dtype=text_embeding.dtype,
            device=text_embeding.device,
        )
        return visual_prompt + self.apply_text_gate(text_embeding)

    def build_structured_waste_prompts(self, text_embedding, image_embedding, attribute_ids=None):
        prompts = self.build_seg_prompt(text_embedding)
        if not (self.use_wcm or self.use_sap or self.use_cgta or self.use_dpd):
            return prompts

        n = prompts.shape[0]
        attr_prior = None
        if self.use_sap and hasattr(self.model, "sap") and attribute_ids is not None:
            attr_ids = attribute_ids.to(device=prompts.device)
            attr_prior, _ = self.model.sap(attr_ids)
            attr_prior = attr_prior.to(dtype=prompts.dtype, device=prompts.device)
            attr_prior = attr_prior.expand(n, -1)
            prompts = prompts + attr_prior.unsqueeze(1)

        if self.use_cgta and hasattr(self.model, "cgta"):
            if self.use_wcm and hasattr(self.model, "wcm"):
                wcm_dtype = next(self.model.wcm.parameters()).dtype
                candidates = self.model.wcm(image_embedding.to(dtype=wcm_dtype))
                candidates = candidates.to(dtype=prompts.dtype, device=prompts.device)
            else:
                candidates = image_embedding.flatten(2).transpose(1, 2).to(dtype=prompts.dtype)
            candidates = candidates.expand(n, -1, -1).contiguous()
            cgta_dtype = next(self.model.cgta.parameters()).dtype
            cgta_attr = attr_prior.to(dtype=cgta_dtype) if attr_prior is not None else None
            prompts = self.model.cgta(
                prompts.to(dtype=cgta_dtype),
                candidates.to(dtype=cgta_dtype),
                attr_prior=cgta_attr,
            ).to(dtype=text_embedding.dtype)

        if self.use_dpd and hasattr(self.model, "dpd"):
            dpd_dtype = next(self.model.dpd.parameters()).dtype
            prompts = self.model.dpd(prompts.to(dtype=dpd_dtype)).to(dtype=text_embedding.dtype)
        return prompts

    def build_final_waste_prompts(
        self,
        multimodal_hidden_states,
        multimodal_attention_mask,
        image_embedding,
        attribute_ids=None,
    ):
        """Build SWQ/SAP/HECG/CAPG prompts for one image."""
        if not (
            self.use_swq_former
            and self.use_hecg
            and self.use_capg
            and hasattr(self.model, "swq_former")
            and hasattr(self.model, "hecg")
            and hasattr(self.model, "capg")
        ):
            raise RuntimeError(
                "Final prompt path requires SWQ-Former, HECG, and CAPG to be enabled."
            )

        swq_dtype = next(self.model.swq_former.parameters()).dtype
        hidden_states = multimodal_hidden_states.to(dtype=swq_dtype)
        padding_mask = None
        if multimodal_attention_mask is not None:
            padding_mask = ~multimodal_attention_mask.bool()
        queries = self.model.swq_former(hidden_states, key_padding_mask=padding_mask)

        attr_weights = None
        if self.use_sap and hasattr(self.model, "sap") and attribute_ids is not None:
            sap_dtype = next(self.model.sap.parameters()).dtype
            attr_prior, attr_weights = self.model.sap(
                attribute_ids.to(device=queries.device),
                context=queries.mean(dim=1).to(dtype=sap_dtype),
            )
            queries = queries + attr_prior.to(dtype=queries.dtype).unsqueeze(1)

        hecg_dtype = next(self.model.hecg.parameters()).dtype
        grounded_queries, evidence_field, consensus_loss, scale_weights = self.model.hecg(
            queries.to(dtype=hecg_dtype),
            image_embedding.to(dtype=hecg_dtype),
        )
        capg_dtype = next(self.model.capg.parameters()).dtype
        prompts, ambiguity, contrast_loss, diversity_loss = self.model.capg(
            grounded_queries.to(dtype=capg_dtype),
            image_embedding.to(dtype=capg_dtype),
            evidence_field.to(dtype=capg_dtype),
        )
        return prompts, {
            "evidence_field": evidence_field,
            "ambiguity": ambiguity,
            "evidence_consensus_loss": consensus_loss,
            "prompt_contrast_loss": contrast_loss,
            "prompt_diversity_loss": diversity_loss,
            "attribute_weights": attr_weights,
            "scale_weights": scale_weights,
        }

    @staticmethod
    def _expand_multimodal_attention_mask(attention_mask, hidden_length):
        """Match the LLaVA attention mask to hidden states after image-token insertion."""
        if attention_mask is None:
            return None
        delta = hidden_length - attention_mask.shape[-1]
        if delta > 0:
            prefix = torch.ones(
                attention_mask.shape[0],
                delta,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix, attention_mask], dim=-1)
        elif delta < 0:
            attention_mask = attention_mask[:, :hidden_length]
        return attention_mask

    def predict(self,points=None,boxes=None,masks=None,text_embeding=None,image_embeddings=None,multimask_output=1):
        text_embeding = self.build_structured_waste_prompts(text_embeding, image_embeddings, None)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.model.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=text_embeding,
        )
        sparse_embeddings = sparse_embeddings.to(text_embeding.dtype)
        
        low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        # avg_attention_map =low_res_masks.cpu().to(torch.float32).numpy()
        # avg_attention_map=avg_attention_map.reshape(256,256)
        # attention_map_reshaped = cv2.resize(avg_attention_map,(512,512))
        return low_res_masks,iou_predictions
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        attribute_ids: torch.LongTensor = None,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
      
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )
    
        if inference:
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            output_i = super().forward(
                images=images_clip_extend,
                attention_mask=attention_masks,
                input_ids=input_ids,
                output_hidden_states=True,
            )
            output_hidden_states = output_i.hidden_states
            output = None
   
        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)
           
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_attentions=True,
                output_hidden_states=True,
            )
          
            output_hidden_states = output.hidden_states

        final_prompt_path = self.use_swq_former and self.use_hecg and self.use_capg
        last_multimodal_hidden_state = output_hidden_states[-1]

        if last_multimodal_hidden_state.dim() == 2:
            last_multimodal_hidden_state = last_multimodal_hidden_state.unsqueeze(0)

        if last_multimodal_hidden_state.dim() != 3:
            raise ValueError(
                "Expected final hidden state [B, L, C], got {}".format(
                    tuple(last_multimodal_hidden_state.shape)
                )
            )

        expanded_attention_masks = self._expand_multimodal_attention_mask(
            attention_masks,
            last_multimodal_hidden_state.shape[1],
        )

        pred_embeddings = None
        if not final_prompt_path:
            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(
                self.model.text_hidden_fcs[0](last_multimodal_hidden_state)
            )
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

            pred_embeddings_flat = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [
                    torch.zeros(
                        1,
                        dtype=torch.long,
                        device=seg_token_offset.device,
                    ),
                    seg_token_offset,
                ],
                dim=0,
            )
            seg_token_offset = seg_token_offset[offset]
            pred_embeddings = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings.append(pred_embeddings_flat[start_i:end_i])

        multimask_output = False
        pred_masks = []
        aux_outputs = []
        num_images = len(offset) - 1
        for i in range(num_images):
            attr_i = None
            if attribute_ids is not None:
                attr_i = attribute_ids[i].unsqueeze(0)

            if final_prompt_path:
                start_i, end_i = int(offset[i].item()), int(offset[i + 1].item())
                memory = last_multimodal_hidden_state[start_i:end_i]
                memory_mask = (
                    expanded_attention_masks[start_i:end_i]
                    if expanded_attention_masks is not None
                    else None
                )
                memory = memory.reshape(1, -1, memory.shape[-1])
                if memory_mask is not None:
                    memory_mask = memory_mask.reshape(1, -1)
                prompt_embeddings, prompt_aux = self.build_final_waste_prompts(
                    memory,
                    memory_mask,
                    image_embeddings[i].unsqueeze(0),
                    attr_i,
                )
                aux_outputs.append(prompt_aux)
            else:
                if pred_embeddings[i].shape[0] == 0:
                    # Legacy checkpoints still require a valid prompt even if a
                    # generated answer omitted [SEG].
                    pred_embeddings[i] = self.model.visual_seg_prompt[0].to(
                        dtype=image_embeddings.dtype,
                        device=image_embeddings.device,
                    )
                prompt_embeddings = self.build_structured_waste_prompts(
                    pred_embeddings[i].unsqueeze(1),
                    image_embeddings[i].unsqueeze(0),
                    attr_i,
                )

            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=prompt_embeddings,
            )
            sparse_embeddings = sparse_embeddings.to(prompt_embeddings.dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),        
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=sparse_embeddings,                     
                dense_prompt_embeddings=dense_embeddings,                       
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "evidence_fields": [
                    item["evidence_field"] for item in aux_outputs
                ] if final_prompt_path else None,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = image_embeddings.new_zeros(())
        mask_dice_loss = image_embeddings.new_zeros(())
        mask_soft_iou_loss = image_embeddings.new_zeros(())
        mask_boundary_loss = image_embeddings.new_zeros(())
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            if gt_mask.shape[0] != pred_mask.shape[0]:
                if gt_mask.shape[0] == 1:
                    gt_mask = gt_mask.expand(pred_mask.shape[0], -1, -1)
                elif pred_mask.shape[0] == 1:
                    pred_mask = pred_mask.expand(gt_mask.shape[0], -1, -1)
                else:
                    num_pair_masks = min(gt_mask.shape[0], pred_mask.shape[0])
                    gt_mask = gt_mask[:num_pair_masks]
                    pred_mask = pred_mask[:num_pair_masks]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            if self.mask_loss_type.startswith("focal_"):
                pixel_loss = sigmoid_focal_loss(
                    pred_mask,
                    gt_mask,
                    num_masks=gt_mask.shape[0],
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                )
            else:
                pixel_loss = sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
            mask_bce_loss += pixel_loss * gt_mask.shape[0]
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            if self.soft_iou_loss_weight > 0:
                mask_soft_iou_loss += (
                    soft_iou_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
            if self.boundary_loss_weight > 0:
                mask_boundary_loss += (
                    boundary_dice_loss(
                        pred_mask, gt_mask, num_masks=gt_mask.shape[0]
                    )
                    * gt_mask.shape[0]
                )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_soft_iou_loss = (
            self.soft_iou_loss_weight
            * mask_soft_iou_loss
            / (num_masks + 1e-8)
        )
        mask_boundary_loss = (
            self.boundary_loss_weight
            * mask_boundary_loss
            / (num_masks + 1e-8)
        )
        mask_loss = (
            mask_bce_loss
            + mask_dice_loss
            + mask_soft_iou_loss
            + mask_boundary_loss
        )

        evidence_consensus_loss = mask_loss.new_zeros(())
        prompt_contrast_loss = mask_loss.new_zeros(())
        prompt_diversity_loss = mask_loss.new_zeros(())
        evidence_supervision_loss = mask_loss.new_zeros(())
        if final_prompt_path and aux_outputs:
            evidence_consensus_loss = torch.stack(
                [item["evidence_consensus_loss"] for item in aux_outputs]
            ).mean()
            prompt_contrast_loss = torch.stack(
                [item["prompt_contrast_loss"] for item in aux_outputs]
            ).mean()
            prompt_diversity_loss = torch.stack(
                [item["prompt_diversity_loss"] for item in aux_outputs]
            ).mean()
            supervised_evidence_losses = []
            for batch_idx, item in enumerate(aux_outputs):
                evidence = item["evidence_field"].float().clamp(1e-5, 1.0 - 1e-5)
                target = gt_masks[batch_idx][:1].unsqueeze(1).float()
                target = F.interpolate(
                    target,
                    size=evidence.shape[-2:],
                    mode="nearest",
                )
                is_negative = target.sum() == 0
                sample_weight = 2.0 if bool(is_negative.item()) else 1.0
                supervised_evidence_losses.append(
                    sample_weight * F.binary_cross_entropy(evidence, target)
                )
            evidence_supervision_loss = torch.stack(
                supervised_evidence_losses
            ).mean()

        loss = (
            ce_loss
            + mask_loss
            + self.evidence_consensus_loss_weight * evidence_consensus_loss
            + self.evidence_supervision_loss_weight * evidence_supervision_loss
            + self.prompt_contrast_loss_weight * prompt_contrast_loss
            + self.prompt_diversity_loss_weight * prompt_diversity_loss
        )

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_soft_iou_loss": mask_soft_iou_loss,
            "mask_boundary_loss": mask_boundary_loss,
            "mask_loss": mask_loss,
            "text_gate": self.get_text_gate().detach(),
            "evidence_consensus_loss": evidence_consensus_loss,
            "evidence_supervision_loss": evidence_supervision_loss,
            "prompt_contrast_loss": prompt_contrast_loss,
            "prompt_diversity_loss": prompt_diversity_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
        indices=image_token_indices.item()
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_attentions=True,
            )
            # make_attention_map(outputs,input_ids,tokenizer)
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences
            # attention_out=Attention_aggregation(outputs.attentions[-1])
            # attention_out=attention_out[:,-1,indices:indices+256].view(16,16)
            # output_attention=output_attention[-1][:,:,indices:indices+256,indices:indices+256]
            
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )
            hidden_states = [] 
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                point=None
                low_res_masks,iou_predictions=self.predict(points=point,text_embeding=pred_embeddings[i].unsqueeze(1),
                                                                 image_embeddings=image_embeddings[i].unsqueeze(0),multimask_output=multimask_output)
                
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])
        return output_ids, pred_masks

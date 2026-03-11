
# ///////////////////////////////////////////////////////////////////////////////  ArcFace + Hard Negative Mining



import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
import math

class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

        # ===========================================================
        # 1. Hyperparameters
        # ===========================================================
        self.margin = 0.2
        self.eps = 1e-6
        self.nce_T = getattr(opt, 'nce_T', 0.07)
        self.num_hard_negatives = getattr(opt, 'num_hard_negatives', 200)

    def forward(self, feat_q, feat_k):
        """
        feat_q: Query features, shape: (Batch * Num_Patches, Dim)
        feat_k: Key features, shape: (Batch * Num_Patches, Dim)
        """
        # -----------------------------------------------------------
        # 1. L2 Normalize features
        # -----------------------------------------------------------
        feat_q = F.normalize(feat_q, p=2, dim=1)
        feat_k = F.normalize(feat_k, p=2, dim=1)

        # -----------------------------------------------------------
        # 2. Batch dimension handling
        # -----------------------------------------------------------
        batchSize = self.opt.batch_size

        if feat_q.shape[0] % batchSize == 0:
            current_batch_size = batchSize
            npatches = feat_q.shape[0] // batchSize
        else:
            current_batch_size = 1
            npatches = feat_q.shape[0]

        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # ===========================================================
        # 3. Positive pairs: ArcFace + Easy Margin
        # ===========================================================
        # (A) Compute original Cosine Similarity
        cos_theta = F.cosine_similarity(feat_q, feat_k, dim=1)
        cos_theta = cos_theta.view(feat_q.shape[0], 1)

        # (B) Numerical stability clamp
        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)

        # (C) Apply ArcFace logic
        if self.margin > 0.0:
            theta = torch.acos(cos_theta)
            theta_m = theta + self.margin

            # Easy Margin protection strategy
            mask = (theta_m < math.pi)
            l_pos_arc = torch.cos(theta_m)

            # Combine: ArcFace for easy samples, CosFace for hard samples
            l_pos = torch.where(mask, l_pos_arc, cos_theta - self.margin * 0.5)
        else:
            # Fallback to standard NCE if margin is 0
            l_pos = cos_theta

        # ===========================================================
        # 4. Negative Pairs calculation
        # ===========================================================
        feat_q_view = feat_q.view(current_batch_size, npatches, dim)
        feat_k_view = feat_k.view(current_batch_size, npatches, dim)

        l_neg_curbatch = torch.bmm(feat_q_view, feat_k_view.transpose(2, 1))

        # Mask out self-comparisons
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))

        l_neg = l_neg_curbatch.view(-1, npatches)

        # ===========================================================
        # 5. Hard Negative Mining
        # ===========================================================
        l_neg_sorted, _ = torch.sort(l_neg, dim=1, descending=True)

        num_neg = min(self.num_hard_negatives, npatches - 1)
        l_neg_hard = l_neg_sorted[:, :num_neg]

        # ===========================================================
        # 6. Compute Final Loss
        # ===========================================================
        out = torch.cat((l_pos, l_neg_hard), dim=1)
        out = out / self.nce_T

        labels = torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        loss = self.cross_entropy_loss(out, labels)

        return loss






# ///////////////////////////////////////////////////////////  Simple Margin + Hard Negative Mining


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from packaging import version

# class PatchNCELoss(nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#         self.opt = opt

#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

#         # Compatibility for boolean mask across PyTorch versions
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

#         # ===========================================================
#         # Hyperparameters
#         # ===========================================================
#         self.margin = 0.2
#         self.nce_T = getattr(opt, 'nce_T', 0.07)
#         self.num_hard_negatives = getattr(opt, 'num_hard_negatives', 200)

#     def forward(self, feat_q, feat_k):
#         """
#         feat_q: Query features, shape: (Batch * Num_Patches, Dim)
#         feat_k: Key features, shape: (Batch * Num_Patches, Dim)
#         """
#         # -----------------------------------------------------------
#         # 1. L2 Normalize features
#         # -----------------------------------------------------------
#         feat_q = F.normalize(feat_q, p=2, dim=1)
#         feat_k = F.normalize(feat_k, p=2, dim=1)

#         # -----------------------------------------------------------
#         # 2. Batch dimension handling
#         # -----------------------------------------------------------
#         batchSize = self.opt.batch_size

#         # Infer current batch size and number of patches
#         if feat_q.shape[0] % batchSize == 0:
#             current_batch_size = batchSize
#             npatches = feat_q.shape[0] // batchSize
#         else:
#             # Fallback for incomplete batches or test time
#             current_batch_size = 1
#             npatches = feat_q.shape[0]

#         dim = feat_q.shape[1]
#         feat_k = feat_k.detach()

#         # -----------------------------------------------------------
#         # 3. Positive pairs calculation
#         # -----------------------------------------------------------
#         l_pos = F.cosine_similarity(feat_q, feat_k, dim=1)
#         l_pos = l_pos.view(feat_q.shape[0], 1)

#         # -----------------------------------------------------------
#         # 4. Negative pairs calculation
#         # -----------------------------------------------------------
#         feat_q_view = feat_q.view(current_batch_size, npatches, dim)
#         feat_k_view = feat_k.view(current_batch_size, npatches, dim)

#         # Batch matrix multiplication for all patch combinations
#         l_neg_curbatch = torch.bmm(feat_q_view, feat_k_view.transpose(2, 1))

#         # -----------------------------------------------------------
#         # 5. Mask diagonal entries (self-comparisons)
#         # -----------------------------------------------------------
#         diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
#         l_neg_curbatch.masked_fill_(diagonal, -float('inf'))

#         l_neg = l_neg_curbatch.view(-1, npatches)

#         # -----------------------------------------------------------
#         # 6. Hard Negative Mining
#         # -----------------------------------------------------------
#         # Sort in descending order to find the hardest negatives
#         l_neg_sorted, _ = torch.sort(l_neg, dim=1, descending=True)
#         num_neg = min(self.num_hard_negatives, npatches - 1)
#         l_neg_hard = l_neg_sorted[:, :num_neg]

#         # -----------------------------------------------------------
#         # 7. Apply Margin and compute Loss
#         # -----------------------------------------------------------
#         # Numerical stability clamp
#         l_pos = torch.clamp(l_pos, -1.0, 1.0)

#         # Apply margin to positive similarity
#         l_pos = l_pos - self.margin

#         # Concatenate logits: [Positive(1), Hard_Negatives(K)]
#         logits = torch.cat((l_pos, l_neg_hard), dim=1)
#         logits = logits / self.nce_T

#         # Compute CrossEntropyLoss (target is class 0)
#         labels = torch.zeros(logits.size(0), dtype=torch.long, device=feat_q.device)
#         loss = self.cross_entropy_loss(logits, labels)

#         return loss






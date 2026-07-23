import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class TokenFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        """
        Multi-class Focal Loss for Token Classification.
        
        Args:
            alpha (Tensor, optional): A manual rescaling weight given to each class.
                                      Shape should be (num_classes,).
            gamma (float): Focusing parameter. Higher values down-weight easy tokens more.
            ignore_index (int): Specifies a target value that is ignored (e.g., -100 for pad tokens).
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(TokenFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits shape: (batch_size, sequence_length, num_classes)
        # targets shape: (batch_size, sequence_length)
        
        # 1. Flatten tensors for element-wise calculation
        num_classes = logits.size(-1)
        logits = logits.view(-1, num_classes)
        targets = targets.view(-1)

        # 2. Create mask to filter out ignored tokens (like padding or special tokens)
        valid_mask = (targets != self.ignore_index)
        
        # If there are no valid tokens in the batch, return zero loss
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Filter active logits and targets
        active_logits = logits[valid_mask]
        active_targets = targets[valid_mask]

        # 3. Calculate Cross Entropy base probabilities (pt)
        # log_softmax is more numerically stable than softmax
        log_p = F.log_softmax(active_logits, dim=-1)
        
        # Gather the log probabilities of the true target classes
        log_pt = log_p.gather(dim=-1, index=active_targets.unsqueeze(1)).squeeze(1)
        p_t = torch.exp(log_pt)

        # 4. Calculate the Focal Loss modulation factor
        focal_weight = (1 - p_t) ** self.gamma
        loss = -focal_weight * log_pt

        # 5. Apply Alpha class weights if provided
        if self.alpha is not None:
            # Ensure alpha is on the correct device
            self.alpha = self.alpha.to(logits.device)
            # Gather alpha weight for each target token
            alpha_t = self.alpha.gather(dim=0, index=active_targets)
            loss = alpha_t * loss

        # 6. Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class FocalLossTrainer(Trainer):

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.loss_fct = TokenFocalLoss(alpha=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        # Calculate custom loss
        loss = self.loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
import torch 
import torch.nn as nn
import torch.nn.functional as F

#focal loss class, the benefit in using focal loss is to give hard samples bigger loss value, so the model can optimize more on the hard samples
class WeightedMulticlassFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor): 1D tensor of shape (num_classes,) with class weights.
            gamma (float): Focusing parameter to downweight easy examples.
            reduction (str): 'none' | 'mean' | 'sum'.
        """
        super(WeightedMulticlassFocalLoss, self).__init__()
        self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Raw predictions before softmax. Shape: (batch_size, num_classes)
            targets (Tensor): Ground truth integer labels. Shape: (batch_size,)
        """
        # 1. Compute stable log probabilities and standard probabilities
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        
        # 2. Extract values corresponding to true class targets via advanced indexing
        log_p_target = log_p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        p_target = p.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        
        # 3. Calculate Focal Loss modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_target) ** self.gamma
        
        # 4. Map class weights (alpha) dynamically to targets and match runtime device
        self.alpha = self.alpha.to(logits.device)
        alpha_target = self.alpha.gather(dim=0, index=targets)
        
        # 5. Combine components: -alpha * (1 - p_t)^gamma * log(p_t)
        loss = -alpha_target * focal_weight * log_p_target
        
        # 6. Apply final batch reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

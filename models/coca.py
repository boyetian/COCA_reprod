import torch
import torch.nn as nn
import torch.nn.functional as F

class COCA(nn.Module):
    def __init__(self, large_model, small_model, lambda_co_adaptation=1.0, lambda_self_adaptation=1.0):
        super(COCA, self).__init__()
        self.large_model = large_model
        self.small_model = small_model
        self.lambda_co_adaptation = lambda_co_adaptation
        self.lambda_self_adaptation = lambda_self_adaptation

    def forward(self, x):
        # Get predictions from both models
        large_model_logits = self.large_model(x)
        small_model_logits = self.small_model(x)

        # Co-adaptation: Use the small model's confident predictions to guide the large model
        pseudo_labels = torch.argmax(small_model_logits, dim=1)
        co_adaptation_loss = F.cross_entropy(large_model_logits, pseudo_labels)

        # Self-adaptation: Enhance each model's own predictions (e.g., via entropy minimization)
        large_model_self_adaptation_loss = self.entropy_loss(large_model_logits)
        small_model_self_adaptation_loss = self.entropy_loss(small_model_logits)

        # Total loss
        loss = (self.lambda_co_adaptation * co_adaptation_loss +
                self.lambda_self_adaptation * (large_model_self_adaptation_loss + small_model_self_adaptation_loss))

        return large_model_logits, loss

    def entropy_loss(self, logits):
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        return -(p * log_p).sum(dim=1).mean() 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class COCA(nn.Module):
    def __init__(self, anchor_model, aux_model, lr_anchor=0.00025, lr_aux=0.001, momentum=0.9):
        super(COCA, self).__init__()
        self.anchor_model = anchor_model
        self.aux_model = aux_model

        # Setup optimizers for BN layers
        self.optimizer_anchor = self.setup_optimizer(self.anchor_model, lr_anchor, momentum)
        self.optimizer_aux = self.setup_optimizer(self.aux_model, lr_aux, momentum)

        # Learnable scaling factor tau
        self.tau = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer_tau = optim.SGD([self.tau], lr=0.01, momentum=momentum)

    def setup_optimizer(self, model, lr, momentum):
        bn_params = []
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                bn_params.extend([p for p in module.parameters() if p.requires_grad])
        return optim.SGD(bn_params, lr=lr, momentum=momentum)

    def forward(self, x):
        p_a = self.anchor_model(x)
        p_s = self.aux_model(x)
        
        p_e_prime = p_a + (p_s / self.tau.detach())
        T = torch.max(p_e_prime, dim=1, keepdim=True)[0] / torch.max(p_a, dim=1, keepdim=True)[0]
        p_e = p_e_prime / T
        
        return p_e

    def update(self, x):
        # Forward pass
        p_a = self.anchor_model(x)
        p_s = self.aux_model(x)

        # 1. Update tau
        self.optimizer_tau.zero_grad()
        l_s = torch.norm(torch.exp(p_a.detach()) - torch.exp(p_s.detach() / self.tau), p=1)
        l_s.backward()
        self.optimizer_tau.step()
        
        # Clamp tau to be positive
        self.tau.data.clamp_(min=1e-6)

        # 2. Form ensemble prediction
        p_e_prime = p_a + (p_s / self.tau)
        T = torch.max(p_e_prime, dim=1, keepdim=True)[0].detach() / torch.max(p_a, dim=1, keepdim=True)[0].detach()
        p_e = p_e_prime / T

        # 3. Calculate losses
        # Marginal entropy loss
        l_mar = self.entropy_loss(p_e)

        # Cross-model knowledge distillation loss
        l_ckd_a = self.kl_loss(p_a, p_e.detach())
        l_ckd_s = self.kl_loss(p_s, p_e.detach())
        l_ckd = l_ckd_a + l_ckd_s

        # Self-adaptation loss
        l_self_a = self.entropy_loss(p_a)
        l_self_s = self.entropy_loss(p_s)
        l_self = l_self_a + l_self_s

        # Total loss
        loss = l_mar + l_ckd + l_self

        # Update models
        self.optimizer_anchor.zero_grad()
        self.optimizer_aux.zero_grad()
        loss.backward()
        self.optimizer_anchor.step()
        self.optimizer_aux.step()

    def entropy_loss(self, logits):
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        return -(p * log_p).sum(dim=1).mean()

    def kl_loss(self, student_logits, teacher_logits):
        return F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1), reduction='batchmean')


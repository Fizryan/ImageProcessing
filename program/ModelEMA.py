import torch
import torch.nn as nn
from typing import Dict, Any

from program.LoggingManager import LoggingManager

logger = LoggingManager.setup_logging(__name__)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        self._one_minus_decay = 1.0 - decay

        self.register()
        logger.info(f"ModelEMA initialized with decay: {decay}")

    def register(self):
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_(param.detach(), alpha=self._one_minus_decay)

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def to(self, device: torch.device):
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)

    def state_dict(self) -> Dict[str, Any]:
        sd = self.model.state_dict()
        for name, param in self.shadow.items():
            if name in sd:
                sd[name] = param
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for name, param in state_dict.items():
            if name in self.shadow:
                self.shadow[name] = param.clone().detach()
                if hasattr(self.model, name):
                    target_device = dict(self.model.named_parameters())[name].device
                    self.shadow[name] = self.shadow[name].to(target_device)

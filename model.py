"""
TMEL-Mamba: Theory-guided Multi-scale Educational Learning Mamba
Core model architecture implementation

Paper: Theory-Guided Multi-Scale Educational Learning Mamba for Cross-Cultural Student Performance Prediction
Authors: Mengting Zhou, Yuchen Zhang*, Liangzheng Lee, Xiangyu Shi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DualChannelFeatureProcessor(nn.Module):
    """
    Theory-guided dual-channel feature processing module
    Based on Multiple Intelligence Theory
    """

    def __init__(self, d_cog: int, d_env: int, d_embed: int = 256):
        super().__init__()
        self.d_cog = d_cog
        self.d_env = d_env
        self.d_embed = d_embed

        # Cognitive features processing
        self.cognitive_mlp = nn.Sequential(
            nn.Linear(d_cog, d_embed),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_embed, d_embed)
        )

        # Environmental features embedding
        self.env_embedding = nn.Embedding(d_env, d_embed)
        self.env_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_embed, d_embed)
        )

        # Theory-guided fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed)
        )

    def forward(self, x_cog: torch.Tensor, x_env: torch.Tensor) -> torch.Tensor:
        # Normalize cognitive features
        z_cog = F.normalize(x_cog, dim=-1)
        z_cog = self.cognitive_mlp(z_cog)

        # Process environmental features
        e_env = self.env_embedding(x_env.long())
        e_env = e_env.mean(dim=-2)  # Average pooling for multi-categorical
        e_env = self.env_mlp(e_env)

        # Theory-guided fusion
        h_fused = torch.cat([z_cog, e_env], dim=-1)
        h_fused = self.fusion_layer(h_fused)

        return h_fused


class MultiScaleDilatedConv(nn.Module):
    """
    Multi-scale dilated convolution feature extraction layer
    Captures temporal patterns at different educational time scales
    """

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.dilation_rates = [1, 2, 4, 8]

        # Parallel dilated convolution branches
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size,
                      padding=(kernel_size - 1) * d // 2, dilation=d)
            for d in self.dilation_rates
        ])

        # Adaptive weight fusion mechanism
        self.attention_weights = nn.Parameter(torch.ones(len(self.dilation_rates)))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.weight_mlp = nn.Linear(d_model, len(self.dilation_rates))

    def forward(self, h_fused: torch.Tensor) -> torch.Tensor:
        # h_fused: [batch_size, seq_len, d_model]
        x = h_fused.transpose(1, 2)  # [batch_size, d_model, seq_len]

        # Apply parallel dilated convolutions
        conv_outputs = []
        for i, conv in enumerate(self.conv_branches):
            conv_out = conv(x)
            conv_outputs.append(conv_out)

        # Dynamic weight computation
        pooled = self.global_pool(x).squeeze(-1)  # [batch_size, d_model]
        alpha = F.softmax(self.weight_mlp(pooled), dim=-1)  # [batch_size, 4]

        # Weighted fusion
        h_multi = torch.zeros_like(conv_outputs[0])
        for i, conv_out in enumerate(conv_outputs):
            h_multi += alpha[:, i:i + 1, None] * conv_out

        return h_multi.transpose(1, 2)  # [batch_size, seq_len, d_model]


class SelectiveStateSpace(nn.Module):
    """
    Theory-guided selective state-space mechanism
    Based on Social Cognitive Theory (Individual-Behavior-Environment)
    """

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Ternary decomposition projections
        self.cognitive_proj = nn.Linear(d_model, d_state)
        self.behavioral_proj = nn.Linear(d_model, d_state)
        self.environmental_proj = nn.Linear(d_model, d_state)

        # Selective parameters
        self.delta_proj = nn.Linear(d_model, d_state)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)

        # Initialize A matrix
        A = torch.randn(d_state, d_state)
        self.A_log = nn.Parameter(torch.log(torch.abs(A) + 1e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Ternary decomposition based on Social Cognitive Theory
        A_cog = torch.sigmoid(self.cognitive_proj(x))
        A_beh = torch.sigmoid(self.behavioral_proj(x))
        A_env = torch.sigmoid(self.environmental_proj(x))

        # Hadamard product for triadic interaction
        A_t = A_cog * A_beh * A_env  # [batch_size, seq_len, d_state]

        # Selective parameters
        delta_t = F.softplus(self.delta_proj(x))
        B_t = self.B_proj(x)
        C_t = self.C_proj(x)

        # State space computation
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            # Discretized state space model
            A_discrete = torch.exp(self.A_log.unsqueeze(0) * delta_t[:, t:t + 1].unsqueeze(-1))
            h = A_discrete.squeeze() * h + B_t[:, t] * x[:, t, :self.d_state]
            y_t = torch.sum(C_t[:, t:t + 1] * h.unsqueeze(1), dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class TMLMambaLayer(nn.Module):
    """
    Single TMEL-Mamba layer with theory-guided components
    """

    def __init__(self, d_model: int, d_state: int = 64, d_ff: int = 512):
        super().__init__()
        self.d_model = d_model

        # Multi-scale dilated convolution
        self.multi_scale_conv = MultiScaleDilatedConv(d_model)

        # Theory-guided selective state space
        self.ssm = SelectiveStateSpace(d_model, d_state)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale feature extraction
        h_multi = self.multi_scale_conv(x)
        h_multi = self.norm1(h_multi + x)

        # Theory-guided state space modeling
        h_ssm = self.ssm(h_multi)
        h_ssm = self.norm2(h_ssm + h_multi)

        # Feed-forward processing
        output = self.ffn(h_ssm)
        output = output + h_ssm

        return output


class TMLMamba(nn.Module):
    """
    Complete TMEL-Mamba model
    Theory-guided Multi-scale Educational Learning Mamba
    """

    def __init__(
            self,
            d_cog: int = 12,
            d_env: int = 8,
            d_model: int = 256,
            d_state: int = 64,
            n_layers: int = 6,
            d_ff: int = 512,
            num_classes: int = 1,
            lambda_theory: float = 0.01,
            lambda_complexity: float = 0.001
    ):
        super().__init__()
        self.d_model = d_model
        self.lambda_theory = lambda_theory
        self.lambda_complexity = lambda_complexity

        # Theory-guided dual-channel feature processing
        self.dual_channel = DualChannelFeatureProcessor(d_cog, d_env, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))

        # Stack of TMEL-Mamba layers
        self.layers = nn.ModuleList([
            TMLMambaLayer(d_model, d_state, d_ff)
            for _ in range(n_layers)
        ])

        # Interpretable prediction layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

        # For attention weight analysis
        self.attention_weights = []

    def forward(self, x_cog: torch.Tensor, x_env: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len = x_cog.shape[0], x_cog.shape[1]

        # Dual-channel feature processing
        h_fused = self.dual_channel(x_cog, x_env)

        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        h = h_fused + pos_enc

        # Process through TMEL-Mamba layers
        layer_outputs = []
        for layer in self.layers:
            h = layer(h)
            layer_outputs.append(h)

        # Global average pooling
        h_global = self.global_pool(h.transpose(1, 2)).squeeze(-1)

        # Final prediction
        predictions = self.predictor(h_global)

        # Calculate importance scores for interpretability
        importance_scores = self._calculate_importance_scores(layer_outputs)

        # Prepare interpretability information
        interpretability_info = {
            'importance_scores': importance_scores,
            'layer_outputs': layer_outputs,
            'global_representation': h_global
        }

        return predictions, interpretability_info

    def _calculate_importance_scores(self, layer_outputs: list) -> dict:
        """Calculate educational dimension importance scores"""
        # Simplified importance calculation
        final_output = layer_outputs[-1]
        seq_len = final_output.shape[1]

        # Calculate attention-like weights
        attention_weights = torch.softmax(final_output.mean(dim=-1), dim=-1)

        # Map to educational dimensions
        cognitive_importance = attention_weights[:, :seq_len // 3].mean(dim=-1)
        behavioral_importance = attention_weights[:, seq_len // 3:2 * seq_len // 3].mean(dim=-1)
        environmental_importance = attention_weights[:, 2 * seq_len // 3:].mean(dim=-1)

        return {
            'cognitive': cognitive_importance,
            'behavioral': behavioral_importance,
            'environmental': environmental_importance
        }

    def compute_theory_loss(self, importance_scores: dict) -> torch.Tensor:
        """Compute theory-guided regularization loss"""
        # Encourage balanced attention across dimensions
        scores = torch.stack([
            importance_scores['cognitive'],
            importance_scores['behavioral'],
            importance_scores['environmental']
        ], dim=1)

        # Variance penalty to encourage balance
        theory_loss = torch.var(scores, dim=1).mean()
        return theory_loss

    def compute_total_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                           importance_scores: dict) -> torch.Tensor:
        """Compute total loss including theory guidance"""
        # Prediction loss (MSE for regression)
        pred_loss = F.mse_loss(predictions.squeeze(), targets)

        # Theory-guided loss
        theory_loss = self.compute_theory_loss(importance_scores)

        # Complexity regularization
        complexity_loss = sum(p.pow(2).sum() for p in self.parameters())

        # Total loss
        total_loss = (pred_loss +
                      self.lambda_theory * theory_loss +
                      self.lambda_complexity * complexity_loss)

        return total_loss


# Training utilities
class TMLMambaTrainer:
    """Training utilities for TMEL-Mamba"""

    def __init__(self, model: TMLMamba, lr: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=150
        )

    def train_step(self, x_cog: torch.Tensor, x_env: torch.Tensor,
                   targets: torch.Tensor) -> dict:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        predictions, interpretability_info = self.model(x_cog, x_env)

        # Compute total loss
        loss = self.model.compute_total_loss(
            predictions, targets, interpretability_info['importance_scores']
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Calculate metrics
        rmse = torch.sqrt(F.mse_loss(predictions.squeeze(), targets))

        return {
            'loss': loss.item(),
            'rmse': rmse.item(),
            'predictions': predictions.detach(),
            'importance_scores': interpretability_info['importance_scores']
        }


# Example usage and model creation
def create_tmel_mamba_model(config: dict = None) -> TMLMamba:
    """Create TMEL-Mamba model with default or custom configuration"""
    if config is None:
        config = {
            'd_cog': 12,  # Number of cognitive features
            'd_env': 8,  # Number of environmental features
            'd_model': 256,  # Model dimension
            'd_state': 64,  # State space dimension
            'n_layers': 6,  # Number of Mamba layers
            'd_ff': 512,  # Feed-forward dimension
            'num_classes': 1,  # Output dimension (regression)
            'lambda_theory': 0.01,  # Theory guidance weight
            'lambda_complexity': 0.001  # Complexity regularization weight
        }

    model = TMLMamba(**config)
    return model


if __name__ == "__main__":
    # Example model creation and forward pass
    model = create_tmel_mamba_model()

    # Example input tensors
    batch_size, seq_len = 32, 50
    x_cog = torch.randn(batch_size, seq_len, 12)  # Cognitive features
    x_env = torch.randint(0, 10, (batch_size, seq_len, 8))  # Environmental features
    targets = torch.randn(batch_size) * 100  # Target scores (0-100)

    # Forward pass
    predictions, interpretability_info = model(x_cog, x_env)

    print(f"Model created successfully!")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Cognitive importance: {interpretability_info['importance_scores']['cognitive'].mean():.4f}")
    print(f"Behavioral importance: {interpretability_info['importance_scores']['behavioral'].mean():.4f}")
    print(f"Environmental importance: {interpretability_info['importance_scores']['environmental'].mean():.4f}")
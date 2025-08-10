"""
Model architectures for the phage-set-transformer package.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MAB(nn.Module):
    """Multi-head Attention Block with configurable parameters."""
    
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, 
                 ln: bool = False, temperature: float = 0.1):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.temperature = temperature
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln = nn.LayerNorm(dim_V) if ln else nn.Identity()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                return_attn: bool = False, chunk_size: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with optional chunking for large sets.
        
        Args:
            Q: Query tensor [B, n_q, d]
            K: Key tensor [B, n_k, d]
            mask: Boolean mask [B, n_k] (True = real, False = pad)
            return_attn: Whether to return attention weights
            chunk_size: If not None, process attention in chunks to save memory
        
        Returns:
            Output tensor, optionally with attention weights
        """
        Q_ = self.fc_q(Q)
        K_ = self.fc_k(K)
        V_ = self.fc_v(K)

        B, n_q, _ = Q_.shape
        _, n_k, _ = K_.shape
        dim_split = self.dim_V // self.num_heads

        # Reshape
        Q_ = Q_.view(B, n_q, self.num_heads, dim_split).transpose(1, 2)  # [B,h,n_q,d_h]
        K_ = K_.view(B, n_k, self.num_heads, dim_split).transpose(1, 2)
        V_ = V_.view(B, n_k, self.num_heads, dim_split).transpose(1, 2)

        # For large sets, compute attention in chunks to save memory
        if chunk_size is not None and n_q > chunk_size:
            outputs = []
            attentions = []

            for i in range(0, n_q, chunk_size):
                end_idx = min(i + chunk_size, n_q)
                Q_chunk = Q_[:, :, i:end_idx, :]

                # Compute attention for this chunk
                A_chunk = torch.matmul(Q_chunk, K_.transpose(-2, -1)) / (np.sqrt(dim_split) * self.temperature)

                # Apply mask if provided
                if mask is not None:
                    mask_expanded = mask.unsqueeze(1).unsqueeze(2)
                    A_chunk = A_chunk.masked_fill(~mask_expanded, float('-inf'))

                A_chunk = torch.softmax(A_chunk, dim=-1)

                # Weighted sum for this chunk
                O_chunk = torch.matmul(A_chunk, V_)
                outputs.append(O_chunk)

                if return_attn:
                    attentions.append(A_chunk)

            # Concatenate chunks
            O = torch.cat(outputs, dim=2)
            if return_attn:
                A = torch.cat(attentions, dim=2)
        else:
            # Compute attention logits with temperature scaling
            A = torch.matmul(Q_, K_.transpose(-2, -1)) / (np.sqrt(dim_split) * self.temperature)  # [B,h,n_q,n_k]

            # Apply mask if provided
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,n_k]
                A = A.masked_fill(~mask_expanded, float('-inf'))

            A = torch.softmax(A, dim=-1)  # [B,h,n_q,n_k]

            # Weighted sum
            O = torch.matmul(A, V_)  # [B,h,n_q,d_h]

        O = O.transpose(1, 2).contiguous().view(B, -1, self.dim_V)
        O = self.ln(O + F.relu(self.fc_o(O)))

        if return_attn:
            return O, A
        else:
            return O


class ISAB(nn.Module):
    """Induced Set Attention Block with configurable parameters and chunking support."""
    
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, 
                 ln: bool = False, temperature: float = 0.1, chunk_size: Optional[int] = None):
        super(ISAB, self).__init__()
        self.num_inds = num_inds
        self.chunk_size = chunk_size
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        self.mab1 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, temperature=temperature)
        self.mab2 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, temperature=temperature)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        if return_attn:
            # Get both attentions in single pass
            H, mab1_attn = self.mab1(self.I.repeat(X.size(0), 1, 1), X, mask=mask, return_attn=True)
            H, mab2_attn = self.mab2(X, H, chunk_size=self.chunk_size, return_attn=True)
            
            # Store both for access via separate method
            self._last_mab1_attn = mab1_attn
            self._last_mab2_attn = mab2_attn
            
            return H, mab2_attn
        else:
            H = self.mab1(self.I.repeat(X.size(0), 1, 1), X, mask=mask, chunk_size=None)
            H = self.mab2(X, H, chunk_size=self.chunk_size)
            return H

    def get_detailed_attention(self):
        """Get both MAB attention weights from last forward pass."""
        return {
            'inducing_to_genes': getattr(self, '_last_mab1_attn', None),
            'genes_to_inducing': getattr(self, '_last_mab2_attn', None)
        }

class PMA(nn.Module):
    """Pooling by Multihead Attention with configurable parameters."""
    
    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, 
                 ln: bool = False, temperature: float = 0.1):
        super(PMA, self).__init__()
        self.num_seeds = num_seeds
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, temperature=temperature)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attn: bool = False) -> torch.Tensor:
        B = X.size(0)
        S = self.S.repeat(B, 1, 1)
        
        if return_attn:
            O, attn = self.mab(S, X, mask=mask, return_attn=True)
            return O, attn
        else:
            O = self.mab(S, X, mask=mask)
            return O


class CrossAttention(nn.Module):
    """Cross-attention between two sets with configurable parameters."""
    
    def __init__(self, dim: int, num_heads: int, ln: bool = False, temperature: float = 0.1):
        super(CrossAttention, self).__init__()
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, temperature=temperature)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                return_attn: bool = False):
        return self.mab(X, Y, mask=mask, return_attn=return_attn)


class FlexibleSetEncoder(nn.Module):
    """An enhanced Set Encoder with configurable number of ISAB layers and chunking support."""
    
    def __init__(self, dim_input: int, dim_output: int, num_heads: int, num_inds: int,
                 num_layers: int = 2, ln: bool = False, temperature: float = 0.1, 
                 chunk_size: Optional[int] = None):
        super(FlexibleSetEncoder, self).__init__()

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(ISAB(dim_input, dim_output, num_heads, num_inds, ln=ln,
                                temperature=temperature, chunk_size=chunk_size))

        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(ISAB(dim_output, dim_output, num_heads, num_inds, ln=ln,
                                    temperature=temperature, chunk_size=chunk_size))

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attn: bool = False) -> torch.Tensor:
        H = X
        last_layer_attn = None
        
        for i, layer in enumerate(self.layers):
            if return_attn and i == len(self.layers) - 1:  # Only last layer attention
                H, last_layer_attn = layer(H, mask=mask, return_attn=True)
            else:
                H = layer(H, mask=mask, return_attn=False)
        
        if return_attn:
            return H, last_layer_attn  # BACKWARDS COMPATIBLE: single tensor
        else:
            return H

    def get_detailed_attention(self):
        """Get detailed attention from last layer including inducing point associations."""
        if hasattr(self.layers[-1], 'get_detailed_attention'):
            return self.layers[-1].get_detailed_attention()
        return None

class ResidualBlock(nn.Module):
    """Single residual block: H = H + (LayerNorm→Linear→Activation→Dropout)(H)"""
    def __init__(self, dim: int, dropout: float, ln: bool, activation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),                                    # Same dimension
            nn.LayerNorm(dim) if ln else nn.Identity(),
            activation,
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.block(x)  # RESIDUAL: input + transformation

class FlexibleStrainPhageTransformer(nn.Module):
    """
    Enhanced strain-phage transformer optimized for large sets with 384-dim embeddings.
    """
    
    def __init__(self,
                embedding_dim: int = 384,
                hidden_dim: int = 512,
                num_heads: int = 8,
                strain_inds: int = 256,
                phage_inds: int = 256,
                num_isab_layers: int = 2,
                num_seeds: int = 1,
                dropout: float = 0.1,
                ln: bool = True,
                temperature: float = 0.1,
                use_cross_attention: bool = True,
                classifier_hidden_layers: int = 1,
                classifier_hidden_dim: Optional[int] = 512,
                activation_function: str = "gelu",
                chunk_size: int = 128,
                normalization_type: str = "none",
                use_residual_classifier: bool = False):  # ADD THIS LINE
        super().__init__()

        # Save parameters
        self.temperature = temperature
        self.use_cross_attention = use_cross_attention
        self.chunk_size = chunk_size
        self.normalization_type = normalization_type

        # Set hidden dim for classifier
        if classifier_hidden_dim is None:
            classifier_hidden_dim = hidden_dim

        # Determine activation function
        if activation_function == "relu":
            self.activation = nn.ReLU()
        elif activation_function == "gelu":
            self.activation = nn.GELU()
        elif activation_function == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
        
        # Input normalization
        if normalization_type == "layer_norm":
            self.input_norm = nn.LayerNorm(embedding_dim)
        else:
            self.input_norm = None

        # Dimension expansion layers for 384-dim to hidden_dim
        self.strain_expand = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()
        self.phage_expand = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()

        # Set Transformer encoders
        self.strain_encoder = FlexibleSetEncoder(
            dim_input=hidden_dim,
            dim_output=hidden_dim,
            num_heads=num_heads,
            num_inds=strain_inds,
            num_layers=num_isab_layers,
            ln=ln,
            temperature=temperature,
            chunk_size=chunk_size
        )
        self.phage_encoder = FlexibleSetEncoder(
            dim_input=hidden_dim,
            dim_output=hidden_dim,
            num_heads=num_heads,
            num_inds=phage_inds,
            num_layers=num_isab_layers,
            ln=ln,
            temperature=temperature,
            chunk_size=chunk_size
        )

        # Cross-attention blocks
        if use_cross_attention:
            self.strain_to_phage = CrossAttention(hidden_dim, num_heads, ln=ln, temperature=temperature)
            self.phage_to_strain = CrossAttention(hidden_dim, num_heads, ln=ln, temperature=temperature)

        # PMA for final pooling
        self.strain_pma = PMA(hidden_dim, num_heads, num_seeds=num_seeds, ln=ln, temperature=temperature)
        self.phage_pma = PMA(hidden_dim, num_heads, num_seeds=num_seeds, ln=ln, temperature=temperature)

        # Classifier
        classifier_layers = []
        input_dim = hidden_dim * num_seeds * 2

        # Input layer
        classifier_layers.append(nn.Linear(input_dim, classifier_hidden_dim))
        classifier_layers.append(nn.LayerNorm(classifier_hidden_dim) if ln else nn.Identity())
        classifier_layers.append(self.activation)
        classifier_layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(classifier_hidden_layers - 1):
            if use_residual_classifier:
                # Use residual block
                classifier_layers.append(
                    ResidualBlock(classifier_hidden_dim, dropout, ln, self.activation)
                )
            else:
                # Regular linear layer
                classifier_layers.append(nn.Linear(classifier_hidden_dim, classifier_hidden_dim))
                classifier_layers.append(nn.LayerNorm(classifier_hidden_dim) if ln else nn.Identity())
                classifier_layers.append(self.activation)
                classifier_layers.append(nn.Dropout(dropout))

        # Output layer
        classifier_layers.append(nn.Linear(classifier_hidden_dim, 1))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self,
                strain_genes: torch.Tensor,
                phage_genes: torch.Tensor,
                strain_mask: Optional[torch.Tensor] = None,
                phage_mask: Optional[torch.Tensor] = None,
                return_attn: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the model.
        
        Args:
            strain_genes: [B, n_s, embedding_dim]
            phage_genes: [B, n_p, embedding_dim]
            strain_mask: Optional boolean mask for strain genes
            phage_mask: Optional boolean mask for phage genes
            return_attn: Whether to return attention weights
            
        Returns:
            logits: [B, 1]
            attention_weights: Optional tuple of attention data if return_attn=True
        """
        # Normalize inputs if specified
        if self.normalization_type == "layer_norm":
            strain_genes = self.input_norm(strain_genes)
            phage_genes = self.input_norm(phage_genes)
        elif self.normalization_type == "l2_norm":
            strain_genes = F.normalize(strain_genes, p=2, dim=-1)
            phage_genes = F.normalize(phage_genes, p=2, dim=-1)

        # Optional dimension expansion
        strain_genes = self.strain_expand(strain_genes)
        phage_genes = self.phage_expand(phage_genes)

        # Encode each genome with set encoders
        if return_attn:
            try:
                strain_enc, strain_encoder_attn = self.strain_encoder(strain_genes, mask=strain_mask, return_attn=True)
                phage_enc, phage_encoder_attn = self.phage_encoder(phage_genes, mask=phage_mask, return_attn=True)
            except TypeError:
                strain_enc = self.strain_encoder(strain_genes, mask=strain_mask)
                phage_enc = self.phage_encoder(phage_genes, mask=phage_mask)
                strain_encoder_attn = phage_encoder_attn = None
        else:
            strain_enc = self.strain_encoder(strain_genes, mask=strain_mask)
            phage_enc = self.phage_encoder(phage_genes, mask=phage_mask)
            strain_encoder_attn = phage_encoder_attn = None

        # Cross-attention (optional)
        strain_attn = None
        phage_attn = None
        
        if self.use_cross_attention:
            if return_attn:
                strain_attended, strain_attn = self.strain_to_phage(strain_enc, phage_enc,
                                                                mask=phage_mask, return_attn=True)
                phage_attended, phage_attn = self.phage_to_strain(phage_enc, strain_enc,
                                                                mask=strain_mask, return_attn=True)
            else:
                strain_attended = self.strain_to_phage(strain_enc, phage_enc, mask=phage_mask)
                phage_attended = self.phage_to_strain(phage_enc, strain_enc, mask=strain_mask)
        else:
            strain_attended = strain_enc
            phage_attended = phage_enc

        # Pool each set
        if return_attn:
            try:
                strain_pooled, strain_pma_attn = self.strain_pma(strain_attended, mask=strain_mask, return_attn=True)
                phage_pooled, phage_pma_attn = self.phage_pma(phage_attended, mask=phage_mask, return_attn=True)
            except TypeError:
                strain_pooled = self.strain_pma(strain_attended, mask=strain_mask)
                phage_pooled = self.phage_pma(phage_attended, mask=phage_mask)
                strain_pma_attn = phage_pma_attn = None
        else:
            strain_pooled = self.strain_pma(strain_attended, mask=strain_mask)
            phage_pooled = self.phage_pma(phage_attended, mask=phage_mask)
            strain_pma_attn = phage_pma_attn = None

        strain_vec = strain_pooled.flatten(1)  # [B, num_seeds*d]
        phage_vec = phage_pooled.flatten(1)    # [B, num_seeds*d]

        # Classifier
        combined = torch.cat([strain_vec, phage_vec], dim=-1)  # [B, 2*num_seeds*d]
        logits = self.classifier(combined)                      # [B, 1]

        if return_attn:
            if self.use_cross_attention:
                # Existing format: (logits, (strain_attn, phage_attn))
                return logits, (strain_attn, phage_attn)
            else:
                # New format for non-cross-attention: pack internal attention as tuple
                internal_attention = {
                    'strain_encoder_attn': strain_encoder_attn,
                    'phage_encoder_attn': phage_encoder_attn,
                    'strain_pma_attn': strain_pma_attn,
                    'phage_pma_attn': phage_pma_attn,
                    # Add access to detailed encoder attention
                    'strain_encoder_detailed': self.strain_encoder.get_detailed_attention(),
                    'phage_encoder_detailed': self.phage_encoder.get_detailed_attention()
                }
                return logits, (internal_attention, None)  # Tuple format maintained
        else:
            return logits


def init_attention_weights(model: nn.Module) -> nn.Module:
    """Initialize weights to encourage attention differentiation."""
    for name, param in model.named_parameters():
        if 'fc_q' in name or 'fc_k' in name:
            # Add small random noise to break symmetry
            with torch.no_grad():
                param.add_(torch.randn_like(param) * 0.1)
    return model

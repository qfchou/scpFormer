"""
ScpFormerConfig: configuration class for the scpFormer model.
"""

from transformers import PretrainedConfig


class ScpFormerConfig(PretrainedConfig):
    """Configuration for ScpFormerModel.

    Attributes:
        d_hid: Hidden dimension of the feedforward network.
        n_embd: Embedding dimension (d_model).
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        dropout: Dropout rate.
        pad_value: Value used for padding in expression vectors.
        use_generative_training: Whether to use generative (perception/generation split) training.
    """

    model_type = "scpFormer"

    def __init__(
        self,
        d_hid=512,
        n_embd=512,
        n_layer=12,
        n_head=8,
        dropout=0.1,
        pad_value=-2,
        use_generative_training=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_hid = d_hid
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.pad_value = pad_value
        self.use_generative_training = use_generative_training

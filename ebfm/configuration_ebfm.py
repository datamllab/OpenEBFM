"""Hugging Face configuration for the canonical EBFM architecture."""

from transformers import PretrainedConfig


EBFM_HEAD_DIM = 128


def _ebfm_depth_scaling(depth, aspect_ratio=64):
    base_dim = depth * aspect_ratio
    embedding_dim = ((base_dim + EBFM_HEAD_DIM - 1) // EBFM_HEAD_DIM) * EBFM_HEAD_DIM
    return {
        "num_transformer_blocks": depth,
        "embedding_dim": embedding_dim,
        "multiheaded_attention_heads": embedding_dim // EBFM_HEAD_DIM,
    }


MODEL_SIZES = {
    f"d{depth}": _ebfm_depth_scaling(depth) for depth in range(4, 29, 2)
}


class EBFMConfig(PretrainedConfig):
    """Configuration persisted in a Transformers model repository."""

    model_type = "ebfm"

    def __init__(
        self,
        model_size="d6",
        vocab_size=32768,
        embedding_dim=None,
        num_transformer_blocks=None,
        multiheaded_attention_heads=None,
        context_length=2048,
        batch_size_per_device=1,
        weight_initialization_method="xavier",
        weight_initialization_gain=1.0,
        ebfm_norm="rms",
        ebfm_act_func="silu",
        ffn_dim_multiplier=None,
        use_sdpa_attention=False,
        float_precision="32-true",
        gradient_checkpointing=False,
        mcmc_step_size=300.0,
        mcmc_step_size_learnable=False,
        mcmc_num_steps=1,
        denoising_initial_condition="random_noise",
        gaussian_random_noise_scaling=1.0,
        free_embed_noise_scale=1.0,
        tf_head_type="direct_unembed",
        **kwargs,
    ):
        if model_size not in MODEL_SIZES:
            valid_sizes = ", ".join(MODEL_SIZES)
            raise ValueError(
                f"unsupported EBFM model_size={model_size!r}; "
                f"use one of: {valid_sizes}"
            )
        size_defaults = MODEL_SIZES[model_size]
        self.model_name = kwargs.pop("model_name", "ebfm")
        self.model_size = model_size
        self.modality = kwargs.pop("modality", "NLP")
        self.time_embedding = kwargs.pop("time_embedding", True)
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim or size_defaults["embedding_dim"])
        self.num_transformer_blocks = int(
            num_transformer_blocks or size_defaults["num_transformer_blocks"]
        )
        self.multiheaded_attention_heads = int(
            multiheaded_attention_heads
            or size_defaults["multiheaded_attention_heads"]
        )
        self.context_length = int(context_length)
        self.batch_size_per_device = int(batch_size_per_device)
        self.weight_initialization_method = weight_initialization_method
        self.weight_initialization_gain = float(weight_initialization_gain)
        self.ebfm_norm = ebfm_norm
        self.ebfm_act_func = ebfm_act_func
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.use_sdpa_attention = bool(use_sdpa_attention)
        self.float_precision = float_precision
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.mcmc_step_size = float(mcmc_step_size)
        self.mcmc_step_size_learnable = bool(mcmc_step_size_learnable)
        self.mcmc_num_steps = int(mcmc_num_steps)
        self.denoising_initial_condition = denoising_initial_condition
        self.gaussian_random_noise_scaling = float(gaussian_random_noise_scaling)
        self.free_embed_noise_scale = float(free_embed_noise_scale)
        self.tf_head_type = tf_head_type
        self.use_tf_head = True
        self.free_embedding_mcmc = True
        self.architectures = kwargs.pop("architectures", ["EBFMForMaskedLM"])
        kwargs.setdefault(
            "auto_map",
            {
                "AutoConfig": "configuration_ebfm.EBFMConfig",
                "AutoModelForMaskedLM": "modeling_ebfm.EBFMForMaskedLM",
                "AutoTokenizer": ["modeling_ebfm.EBFMTokenizer", None],
            },
        )
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

    @classmethod
    def from_legacy_hparams(cls, hparams, vocab_size):
        """Create a config from a Lightning checkpoint hyperparameter map."""

        values = {
            key: hparams[key]
            for key in (
                "model_name",
                "model_size",
                "modality",
                "time_embedding",
                "embedding_dim",
                "num_transformer_blocks",
                "multiheaded_attention_heads",
                "context_length",
                "batch_size_per_device",
                "weight_initialization_method",
                "weight_initialization_gain",
                "ebfm_norm",
                "ebfm_act_func",
                "ffn_dim_multiplier",
                "use_sdpa_attention",
                "float_precision",
                "gradient_checkpointing",
                "mcmc_step_size",
                "mcmc_step_size_learnable",
                "mcmc_num_steps",
                "denoising_initial_condition",
                "gaussian_random_noise_scaling",
                "free_embed_noise_scale",
                "tf_head_type",
            )
            if key in hparams
        }
        values["vocab_size"] = int(vocab_size)
        return cls(**values)

    def to_hparams(self):
        """Return fields required to reconstruct the training model."""

        return {
            key: getattr(self, key)
            for key in (
                "model_name",
                "model_size",
                "modality",
                "time_embedding",
                "vocab_size",
                "embedding_dim",
                "num_transformer_blocks",
                "multiheaded_attention_heads",
                "context_length",
                "batch_size_per_device",
                "weight_initialization_method",
                "weight_initialization_gain",
                "ebfm_norm",
                "ebfm_act_func",
                "ffn_dim_multiplier",
                "use_sdpa_attention",
                "float_precision",
                "gradient_checkpointing",
                "mcmc_step_size",
                "mcmc_step_size_learnable",
                "mcmc_num_steps",
                "denoising_initial_condition",
                "gaussian_random_noise_scaling",
                "free_embed_noise_scale",
                "tf_head_type",
                "use_tf_head",
                "free_embedding_mcmc",
            )
        }


__all__ = ["EBFMConfig", "MODEL_SIZES"]

import torch
import torch.nn as nn
import types
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache

from titans_pytorch.neural_memory import NeuralMemory
from titans_pytorch.memory_models import MemoryMLP, MemoryAttention
from titans_pytorch.mac_transformer import ContinuousAxialPositionalEmbedding, pad_and_segment_with_inverse
from einops import rearrange, repeat, pack, unpack


@dataclass
class QwenMemoryConfig:
    """Configuration class for memory components in Qwen model."""
    # Memory dimensions
    hidden_size: int
    segment_len: int = 128
    num_longterm_mem_tokens: int = 4
    num_persist_mem_tokens: int = 4
    
    # Neural memory configuration
    neural_memory_layers: Tuple[int, ...] = (2, 4, 6, 8, 10)
    neural_memory_segment_len: Optional[int] = None
    neural_mem_gate_attn_output: bool = False
    neural_memory_add_value_residual: bool = False
    neural_memory_qkv_receives_diff_views: bool = True
    neural_mem_weight_residual: bool = False  # Changed to False
    
    # Neural memory architecture
    dim_head: int = 64
    heads: int = 8
    memory_depth: int = 2


class QwenWithIntegratedMemory(nn.Module):
    """
    Qwen model with integrated memory components.
    This implementation adds neural memory to specific decoder layers.
    """
    
    def __init__(
        self, 
        base_model_name_or_path: str,
        memory_config: Optional[QwenMemoryConfig] = None
    ):
        super().__init__()
        
        # Load the base Qwen model
        self.qwen_model = Qwen2ForCausalLM.from_pretrained(base_model_name_or_path)
        config = self.qwen_model.config
        
        # Setup memory configuration
        if memory_config is None:
            memory_config = QwenMemoryConfig(hidden_size=config.hidden_size)
        
        self.memory_config = memory_config
        
        # Ensure neural_memory_segment_len is set
        if self.memory_config.neural_memory_segment_len is None:
            self.memory_config.neural_memory_segment_len = (
                self.memory_config.segment_len + self.memory_config.num_longterm_mem_tokens
            )
            
        # Add long-term memory tokens
        self.longterm_mems = nn.Parameter(
            torch.randn(memory_config.num_longterm_mem_tokens, config.hidden_size) * 0.02
        )
        
        # Add axial positional embeddings for memory segments
        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(
            dim=config.hidden_size, 
            num_axial_dims=2
        )
        
        # Setup neural memory modules
        self.setup_neural_memory()
        
        # Register hooks for memory integration
        self.register_memory_hooks()
        
        # Memory state tracking
        self.reset_memory_state()

    def setup_neural_memory(self):
        """Initialize neural memory components for target layers."""
        self.neural_memories = nn.ModuleDict()
        
        for layer_idx in self.memory_config.neural_memory_layers:
            if layer_idx < len(self.qwen_model.model.layers):
                # Create a neural memory module for this layer
                self.neural_memories[str(layer_idx)] = NeuralMemory(
                    dim=self.memory_config.hidden_size,
                    chunk_size=self.memory_config.neural_memory_segment_len,
                    qkv_receives_diff_views=self.memory_config.neural_memory_qkv_receives_diff_views,
                    dim_head=self.memory_config.dim_head,
                    heads=self.memory_config.heads,
                    momentum=True,
                    qk_rmsnorm=True,
                    accept_weight_residual = False,
                    model=MemoryMLP(
                        dim=self.memory_config.dim_head,
                        depth=self.memory_config.memory_depth,
                        expansion_factor=2.0
                    )
                )
    
    def register_memory_hooks(self):
        """Register forward hooks on target decoder layers."""
        for layer_idx in self.memory_config.neural_memory_layers:
            if layer_idx < len(self.qwen_model.model.layers):
                layer = self.qwen_model.model.layers[layer_idx]
                
                # Using a closure to capture layer_idx
                def make_forward_hook(idx):
                    def hook(module, inputs):
                        hidden_states = inputs[0]
                        
                        # Store hidden state for the QKV layer selector
                        self.mem_input_layers.append(hidden_states)
                        
                        # Apply neural memory
                        if len(self.mem_input_layers) > 1:
                            # Create QKV input from different layers for better memory performance
                            q_input = hidden_states
                            k_input = v_input = self.mem_input_layers[-2]  # Use previous layer
                            qkv_input = torch.stack([q_input, k_input, v_input], dim=0)
                        else:
                            # If not enough layers stored yet, use current hidden states for all
                            qkv_input = torch.stack([hidden_states, hidden_states, hidden_states], dim=0)
                        
                        # Get neural memory for this layer
                        neural_memory = self.neural_memories[str(idx)]
                        
                        # Get current memory state
                        state = self.memory_states.get(idx, None)
                        
                        # Apply neural memory - Never pass prev_weights
                        retrieved, next_state = neural_memory(
                            qkv_input,
                            state=state,
                            prev_weights=None  # Always None
                        )
                        
                        # Store updated state
                        self.memory_states[idx] = next_state
                        
                        # Apply retrieved memory as residual
                        modified_hidden_states = hidden_states + retrieved
                        
                        # Return modified inputs to the module
                        return (modified_hidden_states,) + inputs[1:]
                    
                    return hook
                
                # Register the hook
                layer.register_forward_pre_hook(make_forward_hook(layer_idx))
    
    def reset_memory_state(self):
        """Reset all memory-related state variables."""
        self.memory_states = {}  # Stores NeuralMemoryState for each layer
        self.mem_input_layers = []  # Stores hidden states for QKV selection
        
    def seq_len_with_longterm_mem(self, seq_len):
        """Calculate sequence length after adding memory tokens."""
        return ((seq_len - 1) // self.memory_config.segment_len) * self.memory_config.num_longterm_mem_tokens + seq_len
        
    def prepare_memory_inputs(self, hidden_states):
        """
        Prepare inputs by adding memory tokens and applying positional embeddings.
        This is called at the beginning of the forward pass.
        """
        batch, seq_len = hidden_states.shape[:2]
        
        # Segment the input sequence
        hidden_states, inverse_segment = pad_and_segment_with_inverse(
            hidden_states, 
            self.memory_config.segment_len, 
            inverse_remove_pad=False
        )
        
        # Add long-term memory tokens
        mems = repeat(self.longterm_mems, 'n d -> b n d', b=batch)
        hidden_states, inverse_pack_mems = pack([hidden_states, mems], 'b * d')
        hidden_states = inverse_segment(hidden_states)
        
        # Calculate sequence length with memory
        seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)
        hidden_states = hidden_states[:, :seq_len_with_mem]
        
        # Apply axial positional embeddings
        pos_emb = self.axial_pos_emb.forward_with_seq_len(
            seq_len_with_mem, 
            (self.memory_config.neural_memory_segment_len,)
        )
        hidden_states = hidden_states + pos_emb
        
        return hidden_states
        
    def extract_original_sequence(self, hidden_states, original_seq_len):
        """Extract the original sequence from the sequence with memory tokens."""
        # This is a simplified approach - may need adjustment depending on exact memory token placement
        return hidden_states[:, :original_seq_len]
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Modified forward pass to incorporate memory mechanisms.
        """
        # Reset memory state at the beginning of each forward pass
        self.reset_memory_state()
        
        # Get input embeddings if not provided
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.qwen_model.model.embed_tokens(input_ids)
        
        # Store original sequence length for later extraction
        original_seq_len = inputs_embeds.shape[1]
        
        # Apply memory transformations to input embeddings
        inputs_embeds = self.prepare_memory_inputs(inputs_embeds)
        
        # Forward through the model with modified inputs
        outputs = self.qwen_model(
            input_ids=None,  # We're providing inputs_embeds directly
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        return outputs
    
    def generate(self, *args, **kwargs):
        """Wrapper for the generate method of the base model."""
        # Reset memory states before generation
        self.reset_memory_state()
        return self.qwen_model.generate(*args, **kwargs)


# Utility function to create a Qwen model with integrated memory
def create_qwen_with_memory(
    model_name_or_path: str,
    memory_config: Optional[Dict[str, Any]] = None
) -> QwenWithIntegratedMemory:
    """
    Create a Qwen model with integrated memory components.
    
    Args:
        model_name_or_path: Hugging Face model name or local path
        memory_config: Dictionary of memory configuration parameters
        
    Returns:
        QwenWithIntegratedMemory: The model with memory integration
    """
    # Convert dictionary to QwenMemoryConfig if provided
    memory_cfg = None
    if memory_config is not None:
        memory_cfg = QwenMemoryConfig(**memory_config)
    
    # Create and return the model
    return QwenWithIntegratedMemory(model_name_or_path, memory_cfg)


# Example usage
if __name__ == "__main__":
    # Example configuration
    memory_config = {
        "hidden_size": 896,
        "segment_len": 128,
        "num_longterm_mem_tokens": 0,
        "num_persist_mem_tokens": 0,
        "neural_memory_layers": (2, 4, 6, 8),
        "dim_head": 64,
        "heads": 8,
        "memory_depth": 2,
        "neural_memory_qkv_receives_diff_views": True,
        "neural_mem_weight_residual": False  # Set to False
    }
    
    # Create model
    model = create_qwen_with_memory(
        "Qwen/Qwen2.5-0.5B", 
        memory_config=memory_config
    )
    
    # Example usage
    # inputs = torch.randint(0, 32000, (1, 128))
    inputs = torch.tensor([[151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
            553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
             13, 151645,    198, 151644,    872,    198,    675,    279,   6770,
           6708,    304,  77901,    594,  34360,  30908, 151645,    198, 151644,
          77091,    198]])
    outputs = model(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.logits.shape}")
    print(f"Output logits: {torch.argmax(outputs.logits, dim=-1)}")
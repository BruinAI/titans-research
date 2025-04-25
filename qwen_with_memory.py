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
    
    # Sequence reduction strategy
    # Options: 'conv_pre_dense', 'conv_post_dense', 'drop_second_half', 'drop_random_half'
    sequence_reduction_strategy: str = 'conv_pre_dense'


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
        
        # For sequence reduction strategies
        self.reduction_layers = {}  # Store conv layers by module ID
        self.original_seq_lengths = {}  # Track original sequence lengths by layer
        
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
    
    def make_memory_hook(self, idx):
        """Create a hook that integrates neural memory."""
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
            
            # Store the original sequence length for reduction operations
            self.original_seq_lengths[idx] = hidden_states.shape[1]
            
            # Concatenate retrieved memory with hidden states
            modified_hidden_states = torch.cat([hidden_states, retrieved], dim=1)
            
            # Return modified inputs to the module
            return (modified_hidden_states,) + inputs[1:]
        
        return hook
    
    def make_conv_reduction_hook(self, pre_dense=False):
        """
        Create a hook that applies 1D convolution to reduce sequence length.
        
        Args:
            pre_dense (bool): If True, applies before dense layer; if False, after dense layer
        """
        def hook(module, inputs, outputs):
            # For pre_dense, we intercept after the attention operation
            if pre_dense:
                # In Qwen2, we need to access the attention output
                attn_output = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Dynamically create conv layer if it doesn't exist
                layer_id = id(module)
                if layer_id not in self.reduction_layers:
                    in_dim = attn_output.shape[-1]
                    self.reduction_layers[layer_id] = nn.Conv1d(
                        in_channels=in_dim,
                        out_channels=in_dim,
                        kernel_size=2,
                        stride=2,
                        padding=0
                    ).to(attn_output.device)
                
                # Apply convolution to reduce sequence length
                # Shape: [batch, seq_len, dim] -> [batch, dim, seq_len] -> conv -> [batch, dim, seq_len//2]
                # -> [batch, seq_len//2, dim]
                conv = self.reduction_layers[layer_id]
                x = attn_output.transpose(1, 2)
                x = conv(x)
                x = x.transpose(1, 2)
                
                # Replace in outputs
                if isinstance(outputs, tuple):
                    return (x,) + outputs[1:]
                return x
            
            # For post_dense (after the entire attention module)
            else:
                # Get the hidden state output
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Dynamically create conv layer if it doesn't exist
                layer_id = id(module)
                if layer_id not in self.reduction_layers:
                    in_dim = hidden_states.shape[-1]
                    self.reduction_layers[layer_id] = nn.Conv1d(
                        in_channels=in_dim,
                        out_channels=in_dim,
                        kernel_size=2,
                        stride=2,
                        padding=0
                    ).to(hidden_states.device)
                
                # Apply convolution to reduce sequence length
                conv = self.reduction_layers[layer_id]
                x = hidden_states.transpose(1, 2)
                x = conv(x)
                x = x.transpose(1, 2)
                
                # Replace in outputs
                if isinstance(outputs, tuple):
                    return (x,) + outputs[1:]
                return x
        
        return hook
    
    def make_drop_half_hook(self, random=False):
        """
        Create a hook that drops half of the tokens to reduce sequence length.
        
        Args:
            random (bool): If True, drops random tokens; if False, drops second half
        """
        def hook(module, inputs, outputs):
            # Access the hidden states
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Find the original length (stored when we added the retrieved tokens)
            layer_idx = next((i for i, layer in enumerate(self.qwen_model.model.layers) 
                             if id(layer) == id(module)), None)
            original_len = self.original_seq_lengths.get(layer_idx, hidden_states.shape[1] // 2)
            
            # Drop the second half of tokens
            x = hidden_states[:, :original_len]
            
            # Replace in outputs
            if isinstance(outputs, tuple):
                return (x,) + outputs[1:]
            return x
        
        return hook
    
    def make_drop_random_half_hook(self):
        """Create a hook that randomly drops half of the tokens."""
        def hook(module, inputs, outputs):
            # Access the hidden states
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            batch_size, seq_len, dim = hidden_states.shape
            
            # Find the original length (stored when we added the retrieved tokens)
            layer_idx = next((i for i, layer in enumerate(self.qwen_model.model.layers) 
                             if id(layer) == id(module)), None)
            original_len = self.original_seq_lengths.get(layer_idx, seq_len // 2)
            keep_len = original_len  # Keep tokens equal to original length
            
            # Create random indices for token selection (different for each item in batch)
            indices = []
            for i in range(batch_size):
                # Create random permutation of indices
                perm = torch.randperm(seq_len, device=hidden_states.device)
                # Select the first keep_len indices
                indices.append(perm[:keep_len])
            
            # Stack batch indices
            batch_indices = torch.stack(indices)
            
            # Select tokens batch-wise
            selected_tokens = []
            for i in range(batch_size):
                selected_tokens.append(hidden_states[i, batch_indices[i]])
            
            # Stack back into tensor [batch, keep_len, dim]
            x = torch.stack(selected_tokens)
            
            # Replace in outputs
            if isinstance(outputs, tuple):
                return (x,) + outputs[1:]
            return x
        
        return hook
    
    def register_memory_hooks(self):
        """Register forward hooks on target decoder layers and sequence reduction hooks."""
        for layer_idx in self.memory_config.neural_memory_layers:
            if layer_idx < len(self.qwen_model.model.layers):
                layer = self.qwen_model.model.layers[layer_idx]
                
                # Register the memory hook
                layer.register_forward_pre_hook(self.make_memory_hook(layer_idx))
                
                # Now register the appropriate reduction hook based on the strategy
                strategy = self.memory_config.sequence_reduction_strategy
                
                if strategy == 'conv_pre_dense':
                    # Register hook after attention but before dense layer
                    layer.self_attn.register_forward_hook(self.make_conv_reduction_hook(pre_dense=True))
                elif strategy == 'conv_post_dense':
                    # Register hook after dense layer
                    layer.register_forward_hook(self.make_conv_reduction_hook(pre_dense=False))
                elif strategy == 'drop_second_half':
                    # Register hook to drop second half of tokens
                    layer.register_forward_hook(self.make_drop_half_hook(random=False))
                elif strategy == 'drop_random_half':
                    # Register hook to randomly drop half of tokens
                    layer.register_forward_hook(self.make_drop_random_half_hook())
    
    def reset_memory_state(self, reset=None):
        """
        Reset memory state.
        
        Args:
            reset: Optional boolean whether to reset state between forward calls or not.
                If None or True, resets all states.
                If False, does nothing.
        """
        # Complete reset of all states
        if reset is None or reset is True:
            self.memory_states = {}
            self.mem_input_layers = []
            return
        
        # Don't reset anything
        if reset is False:
            return
        
        
    def seq_len_with_longterm_mem(self, seq_len):
        """Calculate sequence length after adding memory tokens."""
        # Simply add the number of memory tokens to the sequence length
        return seq_len + self.memory_config.num_longterm_mem_tokens
   
    def prepare_memory_inputs(self, hidden_states):
        """
        Prepare inputs by adding memory tokens and applying positional embeddings.
        This function prepends persistent memory tokens to the beginning of each sequence,
        rather than interspersing them between segments.
        """
        batch, seq_len = hidden_states.shape[:2]
        
        # Create memory tokens for each batch
        mems = repeat(self.longterm_mems, 'n d -> b n d', b=batch)
        
        # Simply concatenate memory tokens at the beginning of each sequence
        # This is a simpler approach than interspersing tokens between segments
        hidden_states = torch.cat([mems, hidden_states], dim=1)
        
        # Calculate total sequence length
        seq_len_with_mem = seq_len + self.memory_config.num_longterm_mem_tokens
        
        # Apply axial positional embeddings
        pos_emb = self.axial_pos_emb.forward_with_seq_len(
            seq_len_with_mem, 
            (self.memory_config.neural_memory_segment_len,)
        )
        hidden_states = hidden_states + pos_emb
        
        return hidden_states
    
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
        reset_memory: Optional[bool] = None,
        **kwargs
    ):
        """
        Modified forward pass to incorporate memory mechanisms.
        """
        # Reset memory state at the beginning of each forward pass
        self.reset_memory_state(reset_memory)
        self.original_seq_lengths = {}
        
        # Get input embeddings if not provided
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.qwen_model.model.embed_tokens(input_ids)
        
        # Store original sequence length for later extraction
        original_seq_len = inputs_embeds.shape[1]
        
        # Add persistent memory tokens at the beginning of the sequence
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
        
        # IMPORTANT: We need to remove the memory tokens from the output
        # Since we prepended memory tokens, we need to remove the first num_longterm_mem_tokens
        # positions from the output logits
        if return_dict:
            # For CausalLMOutputWithPast, modify the logits
            num_mem_tokens = self.memory_config.num_longterm_mem_tokens
            outputs.logits = outputs.logits[:, num_mem_tokens:(num_mem_tokens + original_seq_len)]
        else:
            # For tuple outputs, the logits are the first element
            num_mem_tokens = self.memory_config.num_longterm_mem_tokens
            modified_outputs = list(outputs)
            modified_outputs[0] = modified_outputs[0][:, num_mem_tokens:(num_mem_tokens + original_seq_len)]
            outputs = tuple(modified_outputs)
        
        return outputs
    
    def generate(self, *args, reset_memory_state=True, **kwargs):
        """
        Wrapper for the generate method of the base model.
        
        Args:
            reset_memory_state (bool): Whether to reset memory state before generation.
        """
        if reset_memory_state:
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
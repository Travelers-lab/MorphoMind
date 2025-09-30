import torch
from torch import nn
import math
from abc import ABC, abstractmethod


def masked_softmax(x, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(x, dim=-1)
    shape = x.shape
    if valid_lens.dim == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape[-1]
    x.reshape(-1, shape[-1])
    for i in range(x.size(0)):
        x[i, valid_lens[i]:] = 1e-6
    return nn.functional.softmax(x, dim=-1)


class DotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module.

    Implements the core attention mechanism used in Transformer architectures.
    Computes attention weights using dot products between queries and keys,
    then applies these weights to values.

    The attention is scaled by the square root of the key dimension to prevent
    excessively small gradients when the key dimension is large.
    """

    def __init__(self, dropout, **kwargs):
        """
        Initialize the Dot-Product Attention module.

        Args:
            dropout: Dropout probability applied to attention weights
            **kwargs: Additional keyword arguments for base class
        """
        super(DotProductAttention, self).__init__(**kwargs)
        # Dropout layer for attention weights regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Forward pass for scaled dot-product attention.

        Args:
            queries: Query tensor
                Shape: [batch_size, num_queries, d]
            keys: Key tensor
                Shape: [batch_size, num_keys, d]
            values: Value tensor
                Shape: [batch_size, num_values, d]
            valid_lens: Optional valid lengths for sequence masking
                Shape: [batch_size] or [batch_size, num_queries]

        Returns:
            output: Attention-weighted output tensor
                Shape: [batch_size, num_queries, d]

        Processing Steps:
        1. Compute attention scores: Q * K^T / sqrt(d)
        2. Apply masking and softmax to get attention weights
        3. Apply dropout to attention weights
        4. Compute weighted sum: attention_weights * V

        Note:
            - keys and values should have the same sequence length (num_keys == num_values)
            - queries, keys, and values should have the same feature dimension (d)
            - Stores attention_weights as instance variable for visualization/analysis
        """
        # Get feature dimension for scaling
        d = queries.shape[-1]

        # Compute scaled dot-product attention scores
        # scores shape: [batch_size, num_queries, num_keys]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        # Apply masking and softmax to get attention weights
        # attention_weights shape: [batch_size, num_queries, num_keys]
        attention_weights = masked_softmax(scores, valid_lens)

        # Apply attention weights to values and return
        # output shape: [batch_size, num_queries, d]
        return torch.bmm(self.dropout(attention_weights), values)

class AddNorm(nn.Module):
    """
    Add & Normalize module (also known as Residual Connection with Layer Normalization).

    This module implements a key component of Transformer architectures that combines:
    1. Residual connection (adding input to output)
    2. Dropout for regularization
    3. Layer normalization for training stability

    The operation follows the pattern: LayerNorm(X + Dropout(Sublayer(X)))
    This helps with gradient flow and training deep networks.
    """

    def __init__(self, normalized_shape, dropout, **kwargs):
        """
        Initialize the Add & Normalize module.

        Args:
            normalized_shape: Shape for layer normalization
                - If int: features dimension to normalize over
                - If list/tuple: multiple dimensions to normalize over
            dropout: Dropout probability for regularization
            **kwargs: Additional keyword arguments for base class
        """
        super(AddNorm, self).__init__(**kwargs)
        # Layer normalization for stabilizing training
        self.ln = nn.LayerNorm(normalized_shape)
        # Dropout for preventing overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y):
        """
        Forward pass for Add & Normalize operation.

        Args:
            X: Original input tensor (residual connection)
                Shape: [batch_size, seq_len, d_model] or [batch_size, d_model]
            Y: Output tensor from sublayer (attention, FFN, etc.)
                Shape: Must be same as X [batch_size, seq_len, d_model] or [batch_size, d_model]

        Returns:
            output: Normalized and residual-connected output
                Shape: Same as input shape [batch_size, seq_len, d_model] or [batch_size, d_model]

        Processing Steps:
        1. Apply dropout to sublayer output Y
        2. Add residual connection: Dropout(Y) + X
        3. Apply layer normalization to the result

        Note:
            - This implements: LayerNorm(X + Dropout(Sublayer(X)))
            - X and Y must have the same shape
            - Commonly used after attention layers and feed-forward networks in Transformers
        """
        return self.ln(self.dropout(Y) + X)

class PositionWiseFNN(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) module.
    Also known as Position-wise Fully Connected Feed-Forward Network in Transformer architectures.

    This module applies the same feed-forward neural network independently to each position
    in the sequence. It consists of two linear transformations with a GELU activation in between.

    Typically used in Transformer blocks after multi-head attention to add non-linearity
    and increase model capacity.
    """

    def __init__(self, num_inputs, hidden_dim, num_outputs, **kwargs):
        """
        Initialize the Position-wise Feed-Forward Network.

        Args:
            num_inputs: Input dimension size
            hidden_dim: Hidden layer dimension size (usually larger than input/output)
            num_outputs: Output dimension size (usually same as input dimension)
            **kwargs: Additional keyword arguments for base class
        """
        super(PositionWiseFNN, self).__init__(**kwargs)
        # First linear transformation (expansion)
        self.dense1 = nn.Linear(num_inputs, hidden_dim)
        # GELU activation function (commonly used in modern Transformers)
        self.gelu = nn.GELU()
        # Second linear transformation (projection back to original dimension)
        self.dense2 = nn.Linear(hidden_dim, num_outputs)

    def forward(self, X):
        """
        Forward pass for position-wise feed-forward network.

        Args:
            X: Input tensor
                Shape: [batch_size, seq_len, num_inputs] or [batch_size, num_inputs]

        Returns:
            output: Transformed output tensor
                Shape: Same as input shape [batch_size, seq_len, num_outputs] or [batch_size, num_outputs]

        Processing Steps:
        1. First linear transformation: num_inputs -> hidden_dim
        2. GELU activation function
        3. Second linear transformation: hidden_dim -> num_outputs

        Note:
            - The same weights are applied to every position in the sequence
            - Output dimension (num_outputs) is typically the same as input dimension (num_inputs)
            - Hidden dimension (hidden_dim) is typically 2-4 times larger than input dimension
        """
        return self.dense2(self.gelu(self.dense1(X)))


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.transpose(2, 1)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """inverse operation of transpose_qkv"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(2, 1)
    return X.reshape(X.shape[0], X.shape[1], -1)



class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention module.
    Implements the scaled dot-product attention mechanism with multiple attention heads.

    This module performs self-attention or cross-attention on input sequences,
    allowing the model to jointly attend to information from different representation
    subspaces at different positions.
    """

    def __init__(self, num_query, num_key, num_value, hidden_dim, num_heads, dropout, bias=False, **kwargs):
        """
        Initialize the Multi-Head Attention module.

        Args:
            num_query: Input dimension of query vectors
            num_key: Input dimension of key vectors
            num_value: Input dimension of value vectors
            hidden_dim: Hidden dimension size (output dimension)
            num_heads: Number of parallel attention heads
            dropout: Dropout rate for attention weights
            bias: Whether to use bias in linear layers
            **kwargs: Additional keyword arguments for base class
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        # Linear projection layers for queries, keys, and values
        self.W_q = nn.Linear(num_query, hidden_dim, bias=bias)  # Query projection
        self.W_k = nn.Linear(num_key, hidden_dim, bias=bias)  # Key projection
        self.W_v = nn.Linear(num_value, hidden_dim, bias=bias)  # Value projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=bias)  # Output projection

        self.heads = num_heads
        self.attention = DotProductAttention(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Forward pass for multi-head attention.

        Args:
            queries: Query tensor
                Shape: [batch_size, query_seq_len, num_query]
            keys: Key tensor
                Shape: [batch_size, key_seq_len, num_key]
            values: Value tensor
                Shape: [batch_size, value_seq_len, num_value]
            valid_lens: Optional valid lengths for sequence masking
                Shape: [batch_size] or [batch_size, seq_len]

        Returns:
            output: Attention output tensor
                Shape: [batch_size, query_seq_len, hidden_dim]

        Processing Steps:
        1. Linear projection of queries, keys, and values
        2. Reshape for multi-head computation
        3. Apply scaled dot-product attention
        4. Reshape back and apply output projection
        """
        # Linear projections
        queries = transpose_qkv(self.W_q(queries), self.heads)  # [batch_size * heads, query_seq_len, head_dim]
        keys = transpose_qkv(self.W_k(keys), self.heads)  # [batch_size * heads, key_seq_len, head_dim]
        values = transpose_qkv(self.W_v(values), self.heads)  # [batch_size * heads, value_seq_len, head_dim]

        # Prepare valid lengths for masking (if provided)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.heads)

        # Compute attention
        Y = self.attention(queries, keys, values, valid_lens)  # [batch_size * heads, query_seq_len, head_dim]

        # Reshape back to original format
        Y = transpose_output(Y, self.heads)  # [batch_size, query_seq_len, hidden_dim]

        # Final output projection
        return self.W_o(Y)  # [batch_size, query_seq_len, hidden_dim]


class CrossMultiHeadAttention(nn.Module):
    """
    Cross Multi-Head Attention module for processing queries, keys, and values from different sources.
    This module performs attention between sequences of different modalities or representations.

    Key features:
    - Supports different dimensions for queries, keys, and values
    - Handles sequences of different lengths for queries vs keys/values
    - Implements multi-head attention with separate linear projections
    - Includes optional masking for variable-length sequences
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_heads, dropout=0.1, bias=False, **kwargs):
        """
        Initialize the Cross Multi-Head Attention module.

        Args:
            query_dim: Input dimension of query vectors
            key_dim: Input dimension of key vectors
            value_dim: Input dimension of value vectors
            hidden_dim: Hidden dimension size (output dimension)
            num_heads: Number of parallel attention heads
            dropout: Dropout rate for attention weights
            bias: Whether to use bias in linear layers
            **kwargs: Additional keyword arguments
        """
        super(CrossMultiHeadAttention, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        # Validate that hidden dimension is divisible by number of heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Linear projection layers for queries, keys, and values
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=bias)  # Query projection
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=bias)  # Key projection
        self.W_v = nn.Linear(value_dim, hidden_dim, bias=bias)  # Value projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=bias)  # Output projection

        # Dot product attention module
        self.attention = DotProductAttention(dropout)

    def forward(self, queries, keys, values, key_value_valid_lens=None):
        """
        Forward pass for cross multi-head attention.

        Args:
            queries: Query tensor
                Shape: [batch_size, query_seq_len, query_dim]
            keys: Key tensor
                Shape: [batch_size, key_seq_len, key_dim]
            values: Value tensor
                Shape: [batch_size, value_seq_len, value_dim]
            key_value_valid_lens: Optional valid lengths for keys/values masking
                Shape: [batch_size] or [batch_size, key_seq_len]

        Returns:
            output: Attention output tensor
                Shape: [batch_size, query_seq_len, hidden_dim]

        Note:
            - Keys and values must have the same sequence length
            - Queries can have different sequence length than keys/values
            - Output sequence length matches query sequence length
        """
        # Get sequence lengths for validation
        key_seq_len = keys.shape[1]
        value_seq_len = values.shape[1]

        # Ensure keys and values have the same sequence length
        assert key_seq_len == value_seq_len, "Keys and Values must have the same sequence length"

        # Linear projections to hidden dimension
        Q = self.W_q(queries)  # Shape: [batch_size, query_seq_len, hidden_dim]
        K = self.W_k(keys)  # Shape: [batch_size, key_seq_len, hidden_dim]
        V = self.W_v(values)  # Shape: [batch_size, value_seq_len, hidden_dim]

        # Reshape for multi-head attention
        Q_multi = transpose_qkv(Q, self.num_heads)  # [batch_size * num_heads, query_seq_len, head_dim]
        K_multi = transpose_qkv(K, self.num_heads)  # [batch_size * num_heads, key_seq_len, head_dim]
        V_multi = transpose_qkv(V, self.num_heads)  # [batch_size * num_heads, value_seq_len, head_dim]

        # Prepare valid lengths for masking (if provided)
        if key_value_valid_lens is not None:
            valid_lens = torch.repeat_interleave(key_value_valid_lens, self.num_heads)
        else:
            valid_lens = None

        # Compute attention
        attention_output = self.attention(Q_multi, K_multi, V_multi, valid_lens)

        # Reshape back to original format
        attention_output = transpose_output(attention_output, self.num_heads)  # [batch_size, query_seq_len, hidden_dim]

        # Final linear projection
        output = self.W_o(attention_output)

        return output



class MultiModalFusionBlock(nn.Module):
    """
    Multi-modal fusion block that integrates different types of input data.
    This block performs cross-modal attention between identity features and joint data (position/velocity).

    The fusion process involves:
    1. Self-attention on identity features
    2. Cross-attention between identity features and joint data
    3. Feed-forward network with residual connections
    """

    def __init__(self, num_query, num_key, num_value, hidden_dim, num_heads, norm_shape,
                 ffn_num_inputs, ffn_hidden_dim, ffn_num_outputs, dropout, **kwargs):
        """
        Initialize the multi-modal fusion block.

        Args:
            num_query: Dimension of query vectors for attention mechanism
            num_key: Dimension of key vectors for attention mechanism
            num_value: Dimension of value vectors for attention mechanism
            hidden_dim: Hidden dimension size for attention layers
            num_heads: Number of attention heads
            norm_shape: Shape for layer normalization (typically same as hidden dimension)
            ffn_num_inputs: Input dimension for feed-forward network
            ffn_hidden_dim: Hidden dimension for feed-forward network
            ffn_num_outputs: Output dimension for feed-forward network
            dropout: Dropout rate for regularization
            **kwargs: Additional keyword arguments
        """
        super(MultiModalFusionBlock, self).__init__(**kwargs)

        # Cross-attention layer: self-attention on identity features
        self.attention = CrossMultiHeadAttention(num_query, num_key, num_value, hidden_dim, num_heads, dropout)


        # Residual connection with layer normalization
        self.add_norm1 = AddNorm(norm_shape, dropout)
        self.add_norm2 = AddNorm(norm_shape, dropout)

        # Position-wise feed-forward network
        self.ffn = PositionWiseFNN(ffn_num_inputs, ffn_hidden_dim, ffn_num_outputs)
        self.add_layer = nn.Linear(num_query, hidden_dim)

    def forward(self, X, joint_pos, joint_vel):
        """
        Forward pass for multi-modal fusion.

        Args:
            X: Identity features tensor
                Shape: [batch_size, num_identities, num_query]
            joint_pos: Joint position data tensor
                Shape: [batch_size, num_joints, num_key] or [batch_size, num_identities, num_joints, num_key]
            joint_vel: Joint velocity data tensor
                Shape: [batch_size, num_joints, num_value] or [batch_size, num_identities, num_joints, num_value]

        Returns:
            fused_output: Fused multi-modal features
                Shape: [batch_size, num_identities, ffn_num_outputs]

        Processing Steps:
        1. Self-attention on identity features (X -> X -> X)
        2. Residual connection and normalization
        3. Cross-attention between identity features and joint data (Y -> joint_pos -> joint_vel)
        4. Residual connection and normalization
        5. Feed-forward network with residual connection and normalization
        """

        # Cross-attention between identity features and joint data
        # Identity features as query, joint position as key, joint velocity as value
        Y = self.attention(X, joint_pos, joint_vel)
        Y2 = self.add_layer(X)

        # Residual connection and normalization
        Z = self.add_norm1(Y2, Y)

        # Step 3: Feed-forward network with residual connection
        return self.add_norm2(Z, self.ffn(Z))


class BaseInputEncoding(nn.Module, ABC):
    """
    Base class for input encoding modules.
    Provides common functionality for encoding different types of input data.
    """

    def __init__(self, hidden_dim: int, encoding_type: str, **kwargs):
        """
        Initialize the base input encoding module.

        Args:
            hidden_dim: Hidden dimension size for the encoding
            encoding_type: Type of encoding (e.g., 'velocity', 'position', 'acceleration')
            **kwargs: Additional arguments for specific encodings
        """
        super(BaseInputEncoding, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.encoding_type = encoding_type

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for forward pass.

        Args:
            id_tensor: Identity tensor of shape [batch_size, num_id]
            input_tensor: Input data tensor, shape depends on specific encoding

        Returns:
            Encoded features of shape [batch_size, num_id, hidden_dim]
        """
        pass

    def create_weight_matrix(self, num_id: int, num_features: int,
                             init_method: str = 'normal') -> nn.Parameter:
        """
        Create a learnable weight matrix with specified initialization.

        Args:
            num_id: Number of identities
            num_features: Number of input features
            init_method: Weight initialization method ('normal', 'xavier', 'kaiming')

        Returns:
            Learnable weight parameter
        """
        weight = torch.randn(num_id, num_features)

        if init_method == 'xavier':
            nn.init.xavier_uniform_(weight)
        elif init_method == 'kaiming':
            nn.init.kaiming_uniform_(weight)
        elif init_method == 'normal':
            nn.init.normal_(weight, mean=0.0, std=0.02)

        return nn.Parameter(weight)

    def expand_weights_for_batch(self, weights: torch.Tensor, batch_size: int,
                                 num_id: int, num_features: int) -> torch.Tensor:
        """
        Expand weight matrix for batch processing.

        Args:
            weights: Weight tensor of shape [num_id, num_features]
            batch_size: Batch size
            num_id: Number of identities in current batch
            num_features: Number of features

        Returns:
            Expanded weight tensor of shape [batch_size, num_id, num_features, hidden_dim]
        """
        # Select relevant weights based on actual input sizes
        weights_selected = weights[:num_id, :num_features]

        # Expand dimensions for broadcasting
        weights_selected = weights_selected.unsqueeze(-1).expand(
            num_id, num_features, self.hidden_dim
        )

        # Add batch dimension and repeat for batch size
        weights_selected = weights_selected.unsqueeze(0).repeat(
            batch_size, 1, 1, 1
        )

        return weights_selected

    def get_output_shape(self) -> tuple:
        """
        Get the expected output shape.

        Returns:
            Tuple describing output shape [batch_size, num_id, hidden_dim]
        """
        return (-1, -1, self.hidden_dim)


class SelfAttentionBlock(nn.Module):
    """
    Backbone Block module - A customized Transformer-like building block for feature processing.

    This block combines multi-head attention with position-wise feed-forward network,
    using residual connections and layer normalization. It features a unique design
    with an additional linear projection for the residual path.

    Commonly used as a fundamental building block in various neural network architectures
    for tasks requiring sophisticated feature transformation and relationship modeling.
    """

    def __init__(self, num_query, num_key, num_value, hidden_dim, num_heads, norm_shape,
                 ffn_num_inputs, ffn_hidden_dim, ffn_num_outputs, dropout, **kwargs):
        """
        Initialize the Backbone Block.

        Args:
            num_query: Input dimension of query vectors
            num_key: Input dimension of key vectors
            num_value: Input dimension of value vectors
            hidden_dim: Hidden dimension size for attention mechanism
            num_heads: Number of attention heads
            norm_shape: Shape for layer normalization
            ffn_num_inputs: Input dimension for feed-forward network
            ffn_hidden_dim: Hidden dimension for feed-forward network
            ffn_num_outputs: Output dimension for feed-forward network
            dropout: Dropout rate for regularization
            **kwargs: Additional keyword arguments for base class
        """
        super(SelfAttentionBlock, self).__init__(**kwargs)
        # Multi-head attention module
        self.attention = MultiHeadAttention(num_query, num_key, num_value, hidden_dim, num_heads, dropout)

        # Position-wise feed-forward network
        self.fnn = PositionWiseFNN(ffn_num_inputs, ffn_hidden_dim, ffn_num_outputs)

        # Add & Normalize layers for residual connections
        self.add_norm1 = AddNorm(norm_shape, dropout)
        self.add_norm2 = AddNorm(norm_shape, dropout)

        # Additional linear projection for residual connection
        self.add_layer = nn.Linear(num_query, hidden_dim)

    def forward(self, queries, keys, values):
        """
        Forward pass for the Backbone Block.

        Args:
            queries: Query tensor
                Shape: [batch_size, query_seq_len, num_query]
            keys: Key tensor
                Shape: [batch_size, key_seq_len, num_key]
            values: Value tensor
                Shape: [batch_size, value_seq_len, num_value]

        Returns:
            output: Processed output tensor
                Shape: [batch_size, query_seq_len, ffn_num_outputs]

        Processing Steps:
        1. Compute multi-head attention: queries × keys × values
        2. Apply linear projection to original queries for residual connection
        3. First AddNorm: Combine attention output with projected queries
        4. Second AddNorm: Combine previous output with FFN output

        Note:
            - The additional linear projection (add_layer) provides a transformed
              residual path that can help with gradient flow and feature transformation
            - Output dimension is determined by ffn_num_outputs
            - All sequences (queries, keys, values) can have different lengths
        """
        # Step 1: Compute multi-head attention
        output1 = self.attention(queries, keys, values)  # [batch_size, query_seq_len, hidden_dim]

        # Step 2: Apply linear projection to original queries for residual connection
        output2 = self.add_layer(queries)  # [batch_size, query_seq_len, hidden_dim]

        # Step 3: First residual connection with normalization
        # Combines attention output with linearly transformed queries
        output1 = self.add_norm1(output2, output1)  # [batch_size, query_seq_len, hidden_dim]

        # Step 4: Second residual connection with FFN output
        return self.add_norm2(output1, self.fnn(output1))  # [batch_size, query_seq_len, ffn_num_outputs]




if __name__ == "__main__":
    a = torch.normal(1, 2, (2, 225, 1024), device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    b = torch.normal(0, 2, (2, 12, 1024), device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    c = torch.normal(0, 1, (2, 12, 1024), device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    multiAttention = CrossMultiHeadAttention(1024, 1024, 1024, 1024, 8, 0.7)
    multiAttention.cuda()
    multiAttention.eval()
    c = multiAttention(a, b, c)

    # multiAttention = MultiModalFusionBlock(1024, 1024, 1024, 1024,
    #                                  8, [1024], 1024,
    #                                  4096, 1024, 0.7)
    # multiAttention.cuda()
    # multiAttention.eval()
    # c = multiAttention(a, b, c)

    print(c.shape)


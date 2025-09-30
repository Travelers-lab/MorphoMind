import torch
import torch.nn as nn
from core_component import BaseInputEncoding, MultiModalFusionBlock, SelfAttentionBlock



class ESkinEncoding(BaseInputEncoding):
    """
    E-Skin Sensor ID Encoding module for spatiotemporal consistency fusion
    in material point-based robotic systems.

    This module encodes integer sensor IDs from electronic skin systems into
    dense vector representations, specifically designed to prepare sensor
    identification data for spatiotemporal consistency fusion across material
    point collections. The embedding captures both spatial relationships
    between sensor locations and temporal coherence patterns for unified
    representation learning in dynamic robotic environments.
    """

    def __init__(self, num_sensors: int, hidden_dim: int, padding_idx = None,
                 max_norm = None, **kwargs):
        """
        Initialize the E-Skin Sensor ID Encoding module for spatiotemporal fusion.

        Args:
            num_sensors: Total number of unique sensor IDs in the electronic skin system
            hidden_dim: Dimensionality of output embedding vectors for fusion preparation
            padding_idx: Optional index for padding sensors (excluded from gradient updates)
            max_norm: Optional maximum norm for embedding vectors to ensure numerical stability
            **kwargs: Additional arguments for base class initialization
        """
        super(ESkinEncoding, self).__init__(
            hidden_dim=hidden_dim,
            encoding_type='e_skin_spatiotemporal',
            **kwargs
        )

        self.num_sensors = num_sensors
        self.hidden_dim = hidden_dim

        # Core embedding layer for sensor ID to feature vector transformation
        self.embedding = nn.Embedding(
            num_embeddings=num_sensors,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx,
            max_norm=max_norm
        )

        # Spatiotemporal consistency enhancement layers
        self.consistency_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Layer normalization optimized for spatiotemporal fusion
        self.layer_norm = nn.LayerNorm(hidden_dim)


    def forward(self, sensor_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for e-skin sensor ID encoding with spatiotemporal fusion preparation.

        Args:
            sensor_ids: Electronic skin sensor ID tensor for material point fusion
                Shape: [batch_size, n, 1]
                where:
                    - batch_size: Number of material point collections in batch
                    - n: Number of sensor points per collection (material points)
                    - 1: Integer sensor ID for spatiotemporal identification

        Returns:
            encoded_output: Sensor embeddings prepared for spatiotemporal consistency fusion
                Shape: [batch_size, n, hidden_dim]

        Processing Steps:
        1. Embedding lookup for sensor ID to feature space transformation
        2. Spatiotemporal consistency projection for fusion preparation
        3. Adaptive scaling and normalization for stable fusion integration
        4. Output formatting for material point collection processing

        Note:
            - Designed specifically for material point method (MPM) integration
            - Embeddings capture both spatial sensor layout and temporal coherence
            - Output features are optimized for cross-modal spatiotemporal fusion
            - Supports variable numbers of material points (n) per collection
        """
        # Prepare sensor IDs for embedding: [batch_size, n, 1] → [batch_size, n]
        sensor_indices = sensor_ids.squeeze(-1).long()

        # Core embedding: Integer IDs → Dense spatiotemporal features
        embedded = self.embedding(sensor_indices)  # [batch_size, n, hidden_dim]

        # Enhance features for spatiotemporal consistency fusion
        consistency_features = self.consistency_projection(embedded)

        # Apply fusion-optimized normalization
        encoded_output = self.layer_norm(consistency_features)

        return encoded_output


class JointEncoding(BaseInputEncoding):
    """
    Joint-wise Input Encoding module for processing joint-specific data.
    
    This module applies separate linear transformations to each joint in the input data,
    allowing each joint to have its own learned encoding parameters. This is particularly
    useful for robotic systems where different joints may have different characteristics
    and behaviors that require specialized encoding.
    
    The encoding process maintains the joint-wise structure while projecting each joint's
    scalar value into a higher-dimensional hidden space for subsequent processing.
    """

    def __init__(self, num_joints: int, hidden_dim: int, **kwargs):
        """
        Initialize the Joint Encoding module.
        
        Args:
            num_joints: Number of joints to encode (fixed at 12 for this implementation)
            hidden_dim: Hidden dimension size for the encoding
            **kwargs: Additional arguments for base class
        """
        super(JointEncoding, self).__init__(
            hidden_dim=hidden_dim, 
            encoding_type='joint_wise', 
            **kwargs
        )
        
        self.num_joints = num_joints
        
        # Create separate linear encoders for each joint
        # Each joint gets its own nn.Linear(1, hidden_dim) transformation
        self.joint_encoders = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_joints)
        ])
        
        # Optional: Layer normalization for stabilized training
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Optional: Activation function
        self.activation = nn.GELU()

    def forward(self, joint_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for joint-wise encoding.
        
        Args:
            joint_data: Joint input data tensor containing scalar values for each joint
                Shape: [batch_size, num_joints, 1]
                
        Returns:
            encoded_output: Joint-wise encoded features
                Shape: [batch_size, num_joints, hidden_dim]
                
        Processing Steps:
        1. Apply joint-specific linear transformations
        2. Apply activation function (optional)
        3. Apply layer normalization (optional)
        
        Note:
            - Each of the 12 joints is processed by its own dedicated linear layer
            - The identity tensor (id_tensor) is part of the interface but may not be used
              in all implementations
            - Output maintains the same joint dimension (12) but expands feature dimension
        """
        
        batch_size, num_joints, _ = joint_data.shape
        
        # Ensure we have the expected number of joints
        if num_joints != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, but got {num_joints}")
        
        # Process each joint with its dedicated encoder
        encoded_joints = []
        for joint_idx in range(num_joints):
            # Extract data for current joint: [batch_size, 1]
            joint_specific_data = joint_data[:, joint_idx, :]
            
            # Apply joint-specific linear transformation: [batch_size, 1] → [batch_size, hidden_dim]
            joint_encoded = self.joint_encoders[joint_idx](joint_specific_data)
            encoded_joints.append(joint_encoded)
        
        # Stack encoded joints along joint dimension
        # Result shape: [batch_size, num_joints, hidden_dim]
        encoded_output = torch.stack(encoded_joints, dim=1)
        
        # Apply activation function
        encoded_output = self.activation(encoded_output)
        
        # Apply layer normalization for training stability
        encoded_output = self.layer_norm(encoded_output)
        
        return encoded_output


# Spatial encoding modules
class SpatialEncoding(BaseInputEncoding):
    """
    Spatial Input Encoding module for processing 3D spatial data (position, velocity, force).

    This module encodes 3D spatial vectors (x, y, z coordinates) into a higher-dimensional
    hidden space using a two-layer feed-forward network. It is designed to capture spatial
    relationships and patterns in 3D data, making it suitable for robotic applications
    involving spatial coordinates, motion vectors, or force vectors.

    The encoding process transforms 3D spatial information into a rich representation
    that can be effectively processed by subsequent neural network layers.
    """

    def __init__(self, hidden_dim: int, input_dim: int = 3, intermediate_ratio: float = 2.0, encoding_type="spatial", **kwargs):
        """
        Initialize the Spatial Encoding module.

        Args:
            hidden_dim: Hidden dimension size for the final output encoding
            intermediate_ratio: Ratio for intermediate layer size relative to output
                intermediate_size = int(hidden_dim * intermediate_ratio)
            **kwargs: Additional arguments for base class
        """
        super(SpatialEncoding, self).__init__(
            hidden_dim=hidden_dim,
            encoding_type=encoding_type,
            **kwargs
        )

        # Calculate intermediate layer size
        intermediate_size = int(hidden_dim * intermediate_ratio)

        # Two-layer feed-forward network for spatial encoding
        # First layer: 3D input → intermediate dimension
        self.linear1 = nn.Linear(input_dim, intermediate_size)

        # Activation function for non-linearity
        self.activation = nn.GELU()

        # Second layer: intermediate dimension → output hidden dimension
        self.linear2 = nn.Linear(intermediate_size, hidden_dim)

        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Optional dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, spatial_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spatial encoding.

        Args:
            spatial_data: Spatial input data tensor containing 3D vectors
                Shape: [batch_size, n, 3] where:
                    - n: number of spatial points/measurements
                    - 3: (x, y, z) coordinates or spatial components

        Returns:
            encoded_output: Spatially encoded features
                Shape: [batch_size, n, hidden_dim]

        Processing Steps:
        1. First linear transformation: 3D → intermediate dimension
        2. Apply activation function
        3. Second linear transformation: intermediate → hidden dimension
        4. Apply dropout (optional)
        5. Apply layer normalization

        Note:
            - The same encoding weights are applied to all n spatial points
            - The identity tensor (id_tensor) is part of the interface but may not be used
              in all implementations
            - Suitable for encoding 3D position, velocity, or force vectors
            - Maintains the spatial point dimension (n) while expanding feature dimension
        """

        batch_size, num_points, spatial_dim = spatial_data.shape

        # Ensure input has 3 spatial dimensions (x, y, z)
        if spatial_dim != 3:
            raise ValueError(f"Expected 3 spatial dimensions, but got {spatial_dim}")

        # First linear transformation: [batch_size, n, 3] → [batch_size, n, intermediate_size]
        x = self.linear1(spatial_data)

        # Apply activation function
        x = self.activation(x)

        # Second linear transformation: [batch_size, n, intermediate_size] → [batch_size, n, hidden_dim]
        x = self.linear2(x)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Apply layer normalization for training stability
        encoded_output = self.layer_norm(x)

        return encoded_output

class MultiModalFusion(nn.Module):
    """
    Multi-modal Fusion module for cross-attention based integration of material point
    and joint state encodings in robotic systems.

    This module performs hierarchical cross-modal fusion between material point
    encodings (from ESkinEncoding) and joint state encodings (position and velocity
    from JointEncoding) using a 4-layer stack of MultiModalFusionBlock. Each layer
    refines the material point representations by attending to joint state information,
    enabling spatiotemporal consistency in material point system modeling.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 num_layers: int = 4, **kwargs):
        """
        Initialize the Multi-modal Fusion module.

        Args:
            hidden_dim: Hidden dimension size for all input and output features
            num_heads: Number of attention heads in cross-attention layers
            dropout: Dropout rate for regularization
            num_layers: Number of MultiModalFusionBlock layers (default: 4)
            **kwargs: Additional keyword arguments
        """
        super(MultiModalFusion, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create a stack of multi-modal fusion blocks
        self.fusion_layers = nn.ModuleList([
            MultiModalFusionBlock(
                num_query=hidden_dim,
                num_key=hidden_dim,
                num_value=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                norm_shape=hidden_dim,
                ffn_num_inputs=hidden_dim,
                ffn_hidden_dim=hidden_dim * 4,
                ffn_num_outputs=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, material_point_encoding: torch.Tensor,
                joint_position_encoding: torch.Tensor,
                joint_velocity_encoding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-modal fusion of material point and joint encodings.

        Args:
            material_point_encoding: Encoded material point features from ESkinEncoding
                Shape: [batch_size, n, hidden_dim]
            joint_position_encoding: Encoded joint position features from JointEncoding
                Shape: [batch_size, 12, hidden_dim]
            joint_velocity_encoding: Encoded joint velocity features from JointEncoding
                Shape: [batch_size, 12, hidden_dim]

        Returns:
            fused_material_points: Refined material point features after multi-modal fusion
                Shape: [batch_size, n, hidden_dim]

        Processing Steps:
        1. Initialize with original material point encodings as query
        2. For each fusion layer:
           - Use current material points as Q
           - Use joint positions as K (fixed across layers)
           - Use joint velocities as V (fixed across layers)
           - Update material point representations
        3. Return final fused material point features

        Note:
            - Joint position and velocity encodings remain constant across all layers
            - Only material point representations are updated through cross-attention
            - Supports variable numbers of material points (n) while joints are fixed at 12
            - Designed for spatiotemporal consistency in material point system modeling
        """
        # Initialize with input material point encodings
        current_material_points = material_point_encoding

        # Apply multi-layer fusion with joint state information
        for layer in self.fusion_layers:
            # Q: current material points, K: joint positions, V: joint velocities
            current_material_points = layer(
                current_material_points,
                joint_position_encoding,
                joint_velocity_encoding
            )

        return current_material_points


class EncoderBackbone(nn.Module):
    """
    Self-Attention Backbone Encoder for spatial morphology prediction of material point collections.

    This module performs self-attention based encoding on fused material point features
    to predict the spatial morphology of the robotic body and prepare representations
    for the decoder. It consists of multiple self-attention layers that progressively
    refine material point representations through intra-point relationships and
    global context modeling.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 num_layers: int = 6, **kwargs):
        """
        Initialize the Self-Attention Backbone Encoder.

        Args:
            hidden_dim: Hidden dimension size for input and output features
            num_heads: Number of attention heads in self-attention layers
            dropout: Dropout rate for regularization
            num_layers: Number of SelfAttentionBlock layers in the backbone
            **kwargs: Additional keyword arguments
        """
        super(EncoderBackbone, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create a stack of self-attention blocks
        self.self_attention_layers = nn.ModuleList([
            SelfAttentionBlock(
                num_query=hidden_dim,
                num_key=hidden_dim,
                num_value=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                norm_shape=hidden_dim,
                ffn_num_inputs=hidden_dim,
                ffn_hidden_dim=hidden_dim * 4,
                ffn_num_outputs=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, fused_material_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for self-attention backbone encoding.

        Args:
            fused_material_points: Fused material point features from MultiModalFusion
                Shape: [batch_size, n, hidden_dim]
                where:
                    - batch_size: Number of material point collections in batch
                    - n: Number of material points in each collection
                    - hidden_dim: Feature dimension after multi-modal fusion

        Returns:
            encoded_material_points: Self-attention encoded material point features
                Shape: [batch_size, n, hidden_dim]

        Processing Steps:
        1. Initialize with fused material point features as input to first layer
        2. For each self-attention layer:
           - Use current material points as Q, K, V (self-attention)
           - Apply self-attention with residual connections and FFN
           - Update material point representations
        3. Return final encoded material point features for decoder input

        Note:
            - Each layer performs pure self-attention (Q=K=V=material_points)
            - Material point representations are progressively refined through layers
            - Output maintains the same dimensionality for seamless decoder integration
            - Supports variable numbers of material points (n) while preserving relationships
        """
        # Initialize with fused material point features
        current_material_points = fused_material_points

        # Apply multi-layer self-attention encoding
        for layer in self.self_attention_layers:
            # Self-attention: Q, K, V all from current material points
            current_material_points = layer(
                current_material_points,  # queries
                current_material_points,  # keys
                current_material_points  # values
            )

        return current_material_points


class DecoderFusion(nn.Module):
    """
    Decoder Fusion module for cross-attention based integration of spatial encodings
    in the decoder pathway.

    This module performs hierarchical cross-modal fusion between spatial position encodings
    and spatial velocity/force encodings using a 4-layer stack of MultiModalFusionBlock.
    Each layer refines the spatial position representations by attending to velocity and
    force information, enabling dynamic motion modeling and force-aware decoding.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 num_layers: int = 4, **kwargs):
        """
        Initialize the Decoder Fusion module.

        Args:
            hidden_dim: Hidden dimension size for all input and output features
            num_heads: Number of attention heads in cross-attention layers
            dropout: Dropout rate for regularization
            num_layers: Number of MultiModalFusionBlock layers (default: 4)
            **kwargs: Additional keyword arguments
        """
        super(DecoderFusion, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create a stack of multi-modal fusion blocks
        self.fusion_layers = nn.ModuleList([
            MultiModalFusionBlock(
                num_query=hidden_dim,
                num_key=hidden_dim,
                num_value=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                norm_shape=hidden_dim,
                ffn_num_inputs=hidden_dim,
                ffn_hidden_dim=hidden_dim * 4,
                ffn_num_outputs=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, position_encoding: torch.Tensor,
                velocity_encoding: torch.Tensor,
                force_encoding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for decoder fusion of spatial encodings.

        Args:
            position_encoding: Spatial position features from SpatialEncoding
                Shape: [batch_size, n, hidden_dim]
            velocity_encoding: Spatial velocity features from SpatialEncoding
                Shape: [batch_size, n, hidden_dim]
            force_encoding: Spatial force features from SpatialEncoding
                Shape: [batch_size, n, hidden_dim]

        Returns:
            fused_position: Refined position features after multi-modal fusion
                Shape: [batch_size, n, hidden_dim]

        Processing Steps:
        1. Initialize with original position encodings as query
        2. For each fusion layer:
           - Use current position features as Q
           - Use velocity encodings as K (fixed across layers)
           - Use force encodings as V (fixed across layers)
           - Update position representations
        3. Return final fused position features

        Note:
            - Velocity and force encodings remain constant across all layers
            - Only position representations are updated through cross-attention
            - Supports variable spatial points (n) while maintaining feature consistency
            - Designed for dynamic motion modeling with force awareness
        """
        # Initialize with input position encodings
        current_position = position_encoding

        # Apply multi-layer fusion with velocity and force information
        for layer in self.fusion_layers:
            # Q: current position, K: velocity encodings, V: force encodings
            current_position = layer(
                current_position,  # queries (evolving)
                velocity_encoding,  # keys (fixed)
                force_encoding  # values (fixed)
            )

        return current_position


class DecoderBlock(nn.Module):
    """
    Decoder Block module for cross-attention based decoding to predict tracking commands.

    This module performs hierarchical cross-attention between fused target spatial states
    and encoder outputs to generate motion commands for spatial state tracking. It uses
    6 layers of MultiModalFusionBlock to progressively refine target representations
    by attending to encoded spatial context from the encoder pathway.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 num_layers: int = 6, **kwargs):
        """
        Initialize the Decoder Block module.

        Args:
            hidden_dim: Hidden dimension size for all input and output features
            num_heads: Number of attention heads in cross-attention layers
            dropout: Dropout rate for regularization
            num_layers: Number of MultiModalFusionBlock layers (default: 6)
            **kwargs: Additional keyword arguments
        """
        super(DecoderBlock, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create a stack of multi-modal fusion blocks
        self.fusion_layers = nn.ModuleList([
            MultiModalFusionBlock(
                num_query=hidden_dim,
                num_key=hidden_dim,
                num_value=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                norm_shape=hidden_dim,
                ffn_num_inputs=hidden_dim,
                ffn_hidden_dim=hidden_dim * 4,
                ffn_num_outputs=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, fused_target_states: torch.Tensor,
                encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for decoder block cross-attention.

        Args:
            fused_target_states: Fused target spatial states from decoder fusion
                Shape: [batch_size, n, hidden_dim]
                where:
                    - batch_size: Number of spatial state sequences in batch
                    - n: Number of spatial points in each sequence
                    - hidden_dim: Feature dimension after decoder fusion
            encoder_output: Encoded spatial features from encoder backbone
                Shape: [batch_size, n, hidden_dim]
                where:
                    - batch_size: Must match fused_target_states batch size
                    - n: Number of spatial points (must match target states)
                    - hidden_dim: Feature dimension (must match target states)

        Returns:
            decoded_commands: Decoded features for tracking command prediction
                Shape: [batch_size, n, hidden_dim]

        Processing Steps:
        1. Initialize with fused target states as initial query
        2. For each fusion layer:
           - Use current target states as Q (evolving)
           - Use encoder output as K (fixed across layers)
           - Use encoder output as V (fixed across layers)
           - Update target state representations
        3. Return final decoded features for command generation

        Note:
            - Encoder output remains constant across all 6 layers as K and V
            - Target states evolve through layers as Q, becoming encoder-aware
            - Both inputs must have matching batch_size, n, and hidden_dim
            - Output maintains same dimensions for seamless command prediction
        """
        # Initialize with fused target states as query
        current_target = fused_target_states

        # Verify input dimensions match
        if current_target.shape != encoder_output.shape:
            raise ValueError(
                f"Input shapes must match. Target: {current_target.shape}, Encoder: {encoder_output.shape}")

        # Apply 6-layer cross-attention with encoder context
        for layer in self.fusion_layers:
            # Q: current target states, K: encoder output, V: encoder output
            current_target = layer(
                current_target,  # queries (evolving target states)
                encoder_output,  # keys (fixed encoder context)
                encoder_output  # values (fixed encoder context)
            )

        return current_target

# Output decoder module
class OutputDecoder(nn.Module):
    """
    Output Decoder module for transforming transformer backbone outputs into joint commands.

    This module decodes the high-dimensional hidden representations from the transformer
    backbone into specific joint commands for robotic control. It handles the transformation
    from sequence-based representations to fixed-size joint command outputs containing
    position, velocity, and torque values for 12 joints.

    The decoding process involves dimensionality reduction, feature aggregation,
    and specialized projection for each joint command type.
    """

    def __init__(self, input_hidden: int, output_hidden: int = 256, num_joints: int = 12, **kwargs):
        """
        Initialize the Output Decoder module.

        Args:
            input_hidden: Input hidden dimension size from transformer backbone
            output_hidden: Intermediate hidden dimension for processing
            num_joints: Number of joints to control (fixed at 12 for this implementation)
            **kwargs: Additional keyword arguments
        """
        super(OutputDecoder, self).__init__(**kwargs)

        self.input_hidden = input_hidden
        self.output_hidden = output_hidden
        self.num_joints = num_joints

        # Initial projection to reduce sequence dimension and prepare for joint decoding
        self.sequence_projection = nn.Sequential(
            nn.Linear(input_hidden, output_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Global context extraction using attention pooling
        self.context_attention = nn.MultiheadAttention(
            embed_dim=output_hidden,
            num_heads=8,
            batch_first=True
        )

        # Joint-specific decoding branches for position, velocity, and torque
        self.position_decoder = nn.Sequential(
            nn.Linear(output_hidden, output_hidden // 2),
            nn.GELU(),
            nn.Linear(output_hidden // 2, num_joints)  # Output: [batch_size, num_joints]
        )

        self.velocity_decoder = nn.Sequential(
            nn.Linear(output_hidden, output_hidden // 2),
            nn.GELU(),
            nn.Linear(output_hidden // 2, num_joints)  # Output: [batch_size, num_joints]
        )

        self.torque_decoder = nn.Sequential(
            nn.Linear(output_hidden, output_hidden // 2),
            nn.GELU(),
            nn.Linear(output_hidden // 2, num_joints)  # Output: [batch_size, num_joints]
        )

        # Learnable query for attention pooling
        self.context_query = nn.Parameter(torch.randn(1, 1, output_hidden))

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_hidden)

    def forward(self, backbone_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for output decoding.

        Args:
            backbone_output: Output tensor from transformer backbone
                Shape: [batch_size, n, input_hidden]
                where:
                    - batch_size: Number of samples in batch
                    - n: Sequence length (variable number of tokens)
                    - input_hidden: Hidden dimension from backbone

        Returns:
            joint_commands: Decoded joint commands for robotic control
                Shape: [batch_size, 3, 12]
                where:
                    - Dimension 1: 3 command types [position, velocity, torque]
                    - Dimension 2: 12 joints

        Processing Steps:
        1. Project sequence to intermediate hidden dimension
        2. Extract global context using attention pooling
        3. Decode position, velocity, and torque commands in parallel
        4. Stack commands into final output format

        Note:
            - Handles variable sequence length (n) by aggregating global context
            - Each joint command type (position, velocity, torque) has its own decoder branch
            - Output is structured for direct use in robotic control systems
        """
        batch_size, seq_len, hidden_dim = backbone_output.shape

        # Step 1: Project to intermediate dimension
        # [batch_size, n, input_hidden] → [batch_size, n, output_hidden]
        projected_seq = self.sequence_projection(backbone_output)

        # Step 2: Extract global context using attention pooling
        # Create query for all batches
        context_queries = self.context_query.repeat(batch_size, 1, 1)  # [batch_size, 1, output_hidden]

        # Apply attention pooling to get global context
        global_context, _ = self.context_attention(
            query=context_queries,
            key=projected_seq,
            value=projected_seq
        )  # [batch_size, 1, output_hidden]

        # Remove sequence dimension and normalize
        global_context = global_context.squeeze(1)  # [batch_size, output_hidden]
        global_context = self.layer_norm(global_context)

        # Step 3: Decode individual command types in parallel
        position_commands = self.position_decoder(global_context)  # [batch_size, 12]
        velocity_commands = self.velocity_decoder(global_context)  # [batch_size, 12]
        torque_commands = self.torque_decoder(global_context)  # [batch_size, 12]

        # Step 4: Stack commands into final output format
        # Stack along new dimension: [batch_size, 3, 12]
        joint_commands = torch.stack([position_commands, velocity_commands, torque_commands], dim=1)

        return joint_commands


# Main MorphModel
class MorphoModel(nn.Module):
    """
    MorphoModel: A complete encoder-decoder neural network for robotic motion modeling
    and spatial morphology prediction.

    This model integrates multi-modal sensor data from electronic skin and joint states
    through an encoder pathway, then processes spatial commands through a decoder pathway
    to generate precise motion control instructions. The architecture enables spatiotemporal
    consistency modeling for complex robotic systems with material point representations.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, **kwargs):
        """
        Initialize the MorphoModel network.

        Args:
            hidden_dim: Hidden dimension size for all encoding and processing layers
            num_heads: Number of attention heads in multi-head attention layers
            dropout: Dropout rate for regularization throughout the network
            num_encoder_layers: Number of layers in encoder backbone
            num_decoder_layers: Number of layers in decoder backbone
            **kwargs: Additional keyword arguments
        """
        super(MorphoModel, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim

        # Encoder Pathway Components
        # E-skin encoding for tactile sensor data
        self.eskin_encoding = ESkinEncoding(num_sensors=128, hidden_dim=hidden_dim)

        # Joint encodings for position and velocity
        self.joint_position_encoding = JointEncoding(num_joints=12, hidden_dim=hidden_dim)
        self.joint_velocity_encoding = JointEncoding(num_joints=12, hidden_dim=hidden_dim)

        # Multi-modal fusion for encoder
        self.encoder_fusion = MultiModalFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=4
        )

        # Encoder backbone for spatial morphology encoding
        self.encoder_backbone = EncoderBackbone(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_encoder_layers
        )

        # Decoder Pathway Components
        # Spatial encodings for position, velocity, and force
        self.position_encoding = SpatialEncoding(input_dim=3, hidden_dim=hidden_dim)
        self.velocity_encoding = SpatialEncoding(input_dim=3, hidden_dim=hidden_dim)
        self.force_encoding = SpatialEncoding(input_dim=3, hidden_dim=hidden_dim)

        # Decoder fusion for spatial command integration
        self.decoder_fusion = DecoderFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=4
        )

        # Decoder backbone (same structure as encoder backbone)
        self.decoder_backbone = DecoderBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_decoder_layers
        )

        # Output decoder for motion command generation
        self.output_decoder = OutputDecoder(
            input_hidden=hidden_dim,
            output_hidden=2*hidden_dim,
            num_joints=12
        )

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the complete MorphoModel network.

        Args:
            encoder_input: Multi-modal sensor data for encoder pathway
                Shape: [batch_size, n, 25]
                where:
                    - batch_size: Number of samples in batch
                    - n: Number of material points/sensor locations
                    - 25: Feature dimensions [eskin(1) + joint_pos(12) + joint_vel(12)]
            decoder_input: Spatial command data for decoder pathway
                Shape: [batch_size, n, 11]
                where:
                    - batch_size: Must match encoder_input batch_size
                    - n: Number of spatial points (must match encoder_input n)
                    - 11: Feature dimensions [position(3) + velocity(3) + force(3) + padding(2)]

        Returns:
            motion_commands: Predicted joint motion commands for robotic control
                Shape: [batch_size, 3, 12]
                where:
                    - 3: Command types [position, velocity, torque]
                    - 12: Number of robotic joints

        Processing Steps:
        1. Encoder Pathway:
           - ESkin encoding from encoder_input[:,:,0]
           - Joint position encoding from encoder_input[:,:,1:13]
           - Joint velocity encoding from encoder_input[:,:,13:25]
           - Multi-modal fusion of all encoder features
           - Encoder backbone processing

        2. Decoder Pathway:
           - Position encoding from decoder_input[:,:,2:5]
           - Velocity encoding from decoder_input[:,:,5:8]
           - Force encoding from decoder_input[:,:,8:11]
           - Decoder fusion of spatial features
           - Decoder backbone with encoder context
           - Output decoding to motion commands
        """
        batch_size, n, _ = encoder_input.shape

        # Verify input dimensions match
        if encoder_input.shape[0] != decoder_input.shape[0] or encoder_input.shape[1] != decoder_input.shape[1]:
            raise ValueError(f"Encoder and decoder inputs must have matching batch_size and n dimensions")

        # ========== ENCODER PATHWAY ==========

        # ESkin encoding - using first channel
        eskin_features = self.eskin_encoding(
            encoder_input[:, :, 0:1]  # [batch_size, n, 1]
        )

        # Joint position encoding - channels 1-13
        joint_pos_features = self.joint_position_encoding(
            encoder_input[:, 0, 1:13].view(batch_size, 12, 1)  # reshape to [batch_size, 12, 1]
        )

        # Joint velocity encoding - channels 13-25
        joint_vel_features = self.joint_velocity_encoding(
            encoder_input[:, 0, 13:25].view(batch_size, 12, 1)  # reshape to [batch_size, 12, 1]
        )

        # Multi-modal fusion in encoder
        fused_encoder = self.encoder_fusion(
            eskin_features, joint_pos_features, joint_vel_features
        )

        # Encoder backbone processing
        encoder_output = self.encoder_backbone(fused_encoder)

        # ========== DECODER PATHWAY ==========

        # Spatial encodings
        position_features = self.position_encoding(
            decoder_input[:, :, 2:5]  # [batch_size, n, 3]
        )

        velocity_features = self.velocity_encoding(
            decoder_input[:, :, 5:8]  # [batch_size, n, 3]
        )

        force_features = self.force_encoding(
            decoder_input[:, :, 8:11]  # [batch_size, n, 3]
        )

        # Decoder fusion
        fused_decoder = self.decoder_fusion(
            position_features, velocity_features, force_features
        )

        # Decoder backbone with encoder context
        decoder_output = self.decoder_backbone(fused_decoder, encoder_output)

        # Final output decoding to motion commands
        motion_commands = self.output_decoder(decoder_output)

        return motion_commands


# Example usage and testing
if __name__ == "__main__":
    # Test the complete MorphoModel
    batch_size = 4
    n = 32  # number of material points
    hidden_dim = 256

    # Create test data
    tensor = torch.arange(0, 32, dtype=torch.int32)
    integer_channel = tensor.unsqueeze(0).unsqueeze(-1).expand(batch_size, n, 1)
    joint_input = torch.randn(batch_size, n, 24)
    encoder_input = torch.cat([integer_channel, joint_input], dim=-1)
    decoder_input = torch.randn(batch_size, n, 11)

    # Initialize complete model
    morpho_model = MorphoModel(
        hidden_dim=hidden_dim,
        num_heads=8,
        dropout=0.1,
        num_encoder_layers=6,
        num_decoder_layers=6
    )

    # Forward pass
    output = morpho_model(encoder_input, decoder_input)

    print("MorphoModel Test Results:")
    print(f"Encoder input shape: {encoder_input.shape}")  # [4, 32, 25]
    print(f"Decoder input shape: {decoder_input.shape}")  # [4, 32, 11]
    print(f"Output motion commands shape: {output.shape}")  # [4, 3, 12]
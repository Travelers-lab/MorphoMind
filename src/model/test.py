import torch
import torch.nn as nn
import torch.nn.functional as F
from core_component import BaseInputEncoding, MultiModalFusionBlock, SelfAttentionBlock
#
# class SpatialEncoding(BaseInputEncoding):
#     """
#     Spatial Input Encoding module for processing 3D spatial data (position, velocity, force).
#
#     This module encodes 3D spatial vectors (x, y, z coordinates) into a higher-dimensional
#     hidden space using a two-layer feed-forward network. It is designed to capture spatial
#     relationships and patterns in 3D data, making it suitable for robotic applications
#     involving spatial coordinates, motion vectors, or force vectors.
#
#     The encoding process transforms 3D spatial information into a rich representation
#     that can be effectively processed by subsequent neural network layers.
#     """
#
#     def __init__(self, input_dim:int, hidden_dim: int, intermediate_ratio: float = 2.0, encoding_type="spatial", **kwargs):
#         """
#         Initialize the Spatial Encoding module.
#
#         Args:
#             hidden_dim: Hidden dimension size for the final output encoding
#             intermediate_ratio: Ratio for intermediate layer size relative to output
#                 intermediate_size = int(hidden_dim * intermediate_ratio)
#             **kwargs: Additional arguments for base class
#         """
#         super(SpatialEncoding, self).__init__(
#             hidden_dim=hidden_dim,
#             encoding_type=encoding_type,
#             **kwargs
#         )
#
#         # Calculate intermediate layer size
#         intermediate_size = int(hidden_dim * intermediate_ratio)
#
#         # Two-layer feed-forward network for spatial encoding
#         # First layer: 3D input → intermediate dimension
#         self.linear1 = nn.Linear(input_dim, intermediate_size)
#
#         # Activation function for non-linearity
#         self.activation = nn.GELU()
#
#         # Second layer: intermediate dimension → output hidden dimension
#         self.linear2 = nn.Linear(intermediate_size, hidden_dim)
#
#         # Layer normalization for training stability
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#
#         # Optional dropout for regularization
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, id_tensor: torch.Tensor, spatial_data: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for spatial encoding.
#
#         Args:
#             id_tensor: Identity tensor (required by BaseInputEncoding interface)
#                 Shape: [batch_size, num_identities]
#             spatial_data: Spatial input data tensor containing 3D vectors
#                 Shape: [batch_size, n, 3] where:
#                     - n: number of spatial points/measurements
#                     - 3: (x, y, z) coordinates or spatial components
#
#         Returns:
#             encoded_output: Spatially encoded features
#                 Shape: [batch_size, n, hidden_dim]
#
#         Processing Steps:
#         1. Validate input shapes
#         2. First linear transformation: 3D → intermediate dimension
#         3. Apply activation function
#         4. Second linear transformation: intermediate → hidden dimension
#         5. Apply dropout (optional)
#         6. Apply layer normalization
#
#         Note:
#             - The same encoding weights are applied to all n spatial points
#             - The identity tensor (id_tensor) is part of the interface but may not be used
#               in all implementations
#             - Suitable for encoding 3D position, velocity, or force vectors
#             - Maintains the spatial point dimension (n) while expanding feature dimension
#         """
#         # Validate input shapes using base class method
#         self.validate_input_shapes(id_tensor, spatial_data)
#
#         batch_size, num_points, spatial_dim = spatial_data.shape
#
#         # Ensure input has 3 spatial dimensions (x, y, z)
#         if spatial_dim != 3:
#             raise ValueError(f"Expected 3 spatial dimensions, but got {spatial_dim}")
#
#         # First linear transformation: [batch_size, n, 3] → [batch_size, n, intermediate_size]
#         x = self.linear1(spatial_data)
#
#         # Apply activation function
#         x = self.activation(x)
#
#         # Second linear transformation: [batch_size, n, intermediate_size] → [batch_size, n, hidden_dim]
#         x = self.linear2(x)
#
#         # Apply dropout for regularization
#         x = self.dropout(x)
#
#         # Apply layer normalization for training stability
#         encoded_output = self.layer_norm(x)
#
#         return encoded_output
#
#
# # Example usage and testing
# if __name__ == "__main__":
#     # Test the SpatialEncoding module
#     batch_size = 4
#     num_points = 8  # Can be any number of spatial points
#     hidden_dim = 64
#
#     # Create test data
#     id_tensor = torch.randn(batch_size, 6)  # Identity tensor
#     spatial_data = torch.randn(batch_size, num_points, 3)  # 3D spatial data
#
#     # Initialize spatial encoder
#     spatial_encoder = SpatialEncoding(input_dim=3, hidden_dim=hidden_dim, intermediate_ratio=2.0)
#
#     # Forward pass
#     output = spatial_encoder(id_tensor, spatial_data)
#
#     print("SpatialEncoding Test Results:")
#     print(f"Input spatial data shape: {spatial_data.shape}")  # [4, 8, 3]
#     print(f"Output encoded shape: {output.shape}")  # [4, 8, 64]
#     print(f"First linear layer weight shape: {spatial_encoder.linear1.weight.shape}")  # [128, 3]
#     print(f"Second linear layer weight shape: {spatial_encoder.linear2.weight.shape}")  # [64, 128]
#
#     # Test with different number of points
#     spatial_data_variable = torch.randn(batch_size, 15, 3)  # Different number of points
#     output_variable = spatial_encoder(id_tensor, spatial_data_variable)
#     print(f"Variable points input shape: {spatial_data_variable.shape}")  # [4, 15, 3]
#     print(f"Variable points output shape: {output_variable.shape}")  # [4, 15, 64]


# class OutputDecoder(nn.Module):
#     """
#     Output Decoder module for transforming transformer backbone outputs into joint commands.
#
#     This module decodes the high-dimensional hidden representations from the transformer
#     backbone into specific joint commands for robotic control. It handles the transformation
#     from sequence-based representations to fixed-size joint command outputs containing
#     position, velocity, and torque values for 12 joints.
#
#     The decoding process involves dimensionality reduction, feature aggregation,
#     and specialized projection for each joint command type.
#     """
#
#     def __init__(self, input_hidden: int, output_hidden: int = 256, num_joints: int = 12, **kwargs):
#         """
#         Initialize the Output Decoder module.
#
#         Args:
#             input_hidden: Input hidden dimension size from transformer backbone
#             output_hidden: Intermediate hidden dimension for processing
#             num_joints: Number of joints to control (fixed at 12 for this implementation)
#             **kwargs: Additional keyword arguments
#         """
#         super(OutputDecoder, self).__init__(**kwargs)
#
#         self.input_hidden = input_hidden
#         self.output_hidden = output_hidden
#         self.num_joints = num_joints
#
#         # Initial projection to reduce sequence dimension and prepare for joint decoding
#         self.sequence_projection = nn.Sequential(
#             nn.Linear(input_hidden, output_hidden),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )
#
#         # Global context extraction using attention pooling
#         self.context_attention = nn.MultiheadAttention(
#             embed_dim=output_hidden,
#             num_heads=8,
#             batch_first=True
#         )
#
#         # Joint-specific decoding branches for position, velocity, and torque
#         self.position_decoder = nn.Sequential(
#             nn.Linear(output_hidden, output_hidden // 2),
#             nn.GELU(),
#             nn.Linear(output_hidden // 2, num_joints)  # Output: [batch_size, num_joints]
#         )
#
#         self.velocity_decoder = nn.Sequential(
#             nn.Linear(output_hidden, output_hidden // 2),
#             nn.GELU(),
#             nn.Linear(output_hidden // 2, num_joints)  # Output: [batch_size, num_joints]
#         )
#
#         self.torque_decoder = nn.Sequential(
#             nn.Linear(output_hidden, output_hidden // 2),
#             nn.GELU(),
#             nn.Linear(output_hidden // 2, num_joints)  # Output: [batch_size, num_joints]
#         )
#
#         # Learnable query for attention pooling
#         self.context_query = nn.Parameter(torch.randn(1, 1, output_hidden))
#
#         # Layer normalization for stability
#         self.layer_norm = nn.LayerNorm(output_hidden)
#
#     def forward(self, backbone_output: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for output decoding.
#
#         Args:
#             backbone_output: Output tensor from transformer backbone
#                 Shape: [batch_size, n, input_hidden]
#                 where:
#                     - batch_size: Number of samples in batch
#                     - n: Sequence length (variable number of tokens)
#                     - input_hidden: Hidden dimension from backbone
#
#         Returns:
#             joint_commands: Decoded joint commands for robotic control
#                 Shape: [batch_size, 3, 12]
#                 where:
#                     - Dimension 1: 3 command types [position, velocity, torque]
#                     - Dimension 2: 12 joints
#
#         Processing Steps:
#         1. Project sequence to intermediate hidden dimension
#         2. Extract global context using attention pooling
#         3. Decode position, velocity, and torque commands in parallel
#         4. Stack commands into final output format
#
#         Note:
#             - Handles variable sequence length (n) by aggregating global context
#             - Each joint command type (position, velocity, torque) has its own decoder branch
#             - Output is structured for direct use in robotic control systems
#         """
#         batch_size, seq_len, hidden_dim = backbone_output.shape
#
#         # Step 1: Project to intermediate dimension
#         # [batch_size, n, input_hidden] → [batch_size, n, output_hidden]
#         projected_seq = self.sequence_projection(backbone_output)
#
#         # Step 2: Extract global context using attention pooling
#         # Create query for all batches
#         context_queries = self.context_query.repeat(batch_size, 1, 1)  # [batch_size, 1, output_hidden]
#
#         # Apply attention pooling to get global context
#         global_context, _ = self.context_attention(
#             query=context_queries,
#             key=projected_seq,
#             value=projected_seq
#         )  # [batch_size, 1, output_hidden]
#
#         # Remove sequence dimension and normalize
#         global_context = global_context.squeeze(1)  # [batch_size, output_hidden]
#         global_context = self.layer_norm(global_context)
#
#         # Step 3: Decode individual command types in parallel
#         position_commands = self.position_decoder(global_context)  # [batch_size, 12]
#         velocity_commands = self.velocity_decoder(global_context)  # [batch_size, 12]
#         torque_commands = self.torque_decoder(global_context)  # [batch_size, 12]
#
#         # Step 4: Stack commands into final output format
#         # Stack along new dimension: [batch_size, 3, 12]
#         joint_commands = torch.stack([position_commands, velocity_commands, torque_commands], dim=1)
#
#         return joint_commands
#
#
# # Alternative implementation with sequence aggregation
# class OutputDecoderWithAggregation(nn.Module):
#     """
#     Alternative Output Decoder with multiple aggregation strategies.
#
#     This variant provides different methods for aggregating sequence information
#     before decoding joint commands, offering flexibility for different use cases.
#     """
#
#     def __init__(self, input_hidden: int, output_hidden: int = 256, num_joints: int = 12,
#                  aggregation_method: str = "attention", **kwargs):
#         """
#         Initialize alternative output decoder.
#
#         Args:
#             input_hidden: Input hidden dimension size
#             output_hidden: Intermediate hidden dimension
#             num_joints: Number of joints
#             aggregation_method: Method for sequence aggregation
#                 ("attention", "mean", "max", "learned")
#             **kwargs: Additional arguments
#         """
#         super(OutputDecoderWithAggregation, self).__init__(**kwargs)
#
#         self.aggregation_method = aggregation_method
#
#         # Projection layer
#         self.projection = nn.Sequential(
#             nn.Linear(input_hidden, output_hidden),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )
#
#         # Learnable aggregation weights if using learned method
#         if aggregation_method == "learned":
#             self.aggregation_weights = nn.Parameter(torch.ones(1, 1, 1))
#
#         # Command decoders (same as main implementation)
#         self.position_decoder = nn.Sequential(
#             nn.Linear(output_hidden, output_hidden // 2),
#             nn.GELU(),
#             nn.Linear(output_hidden // 2, num_joints)
#         )
#
#         self.velocity_decoder = nn.Sequential(
#             nn.Linear(output_hidden, output_hidden // 2),
#             nn.GELU(),
#             nn.Linear(output_hidden // 2, num_joints)
#         )
#
#         self.torque_decoder = nn.Sequential(
#             nn.Linear(output_hidden, output_hidden // 2),
#             nn.GELU(),
#             nn.Linear(output_hidden // 2, num_joints)
#         )
#
#         self.layer_norm = nn.LayerNorm(output_hidden)
#
#     def forward(self, backbone_output: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass with configurable aggregation.
#         """
#         batch_size, seq_len, hidden_dim = backbone_output.shape
#
#         # Project to intermediate dimension
#         projected = self.projection(backbone_output)  # [batch_size, seq_len, output_hidden]
#
#         # Aggregate sequence information
#         if self.aggregation_method == "attention":
#             # Use the same attention pooling as main implementation
#             query = torch.mean(projected, dim=1, keepdim=True)  # [batch_size, 1, output_hidden]
#             global_context, _ = nn.MultiheadAttention(
#                 output_hidden, 8, batch_first=True
#             )(query, projected, projected)
#             global_context = global_context.squeeze(1)
#
#         elif self.aggregation_method == "mean":
#             global_context = torch.mean(projected, dim=1)  # [batch_size, output_hidden]
#
#         elif self.aggregation_method == "max":
#             global_context = torch.max(projected, dim=1)[0]  # [batch_size, output_hidden]
#
#         elif self.aggregation_method == "learned":
#             weights = F.softmax(self.aggregation_weights.repeat(batch_size, seq_len, 1), dim=1)
#             global_context = torch.sum(projected * weights, dim=1)  # [batch_size, output_hidden]
#
#         global_context = self.layer_norm(global_context)
#
#         # Decode commands
#         position_commands = self.position_decoder(global_context)
#         velocity_commands = self.velocity_decoder(global_context)
#         torque_commands = self.torque_decoder(global_context)
#
#         # Stack into final format
#         joint_commands = torch.stack([position_commands, velocity_commands, torque_commands], dim=1)
#
#         return joint_commands
#
#
# # Example usage and testing
# if __name__ == "__main__":
#     # Test the OutputDecoder module
#     batch_size = 4
#     seq_len = 16  # Variable sequence length
#     input_hidden = 512
#     num_joints = 12
#
#     # Create test data
#     backbone_output = torch.randn(batch_size, seq_len, input_hidden)
#
#     # Initialize output decoder
#     output_decoder = OutputDecoder(input_hidden=input_hidden, output_hidden=256)
#
#     # Forward pass
#     joint_commands = output_decoder(backbone_output)
#
#     print("OutputDecoder Test Results:")
#     print(f"Input backbone output shape: {backbone_output.shape}")  # [4, 16, 512]
#     print(f"Output joint commands shape: {joint_commands.shape}")  # [4, 3, 12]
#     print(f"Position commands range: [{joint_commands[:, 0, :].min():.3f}, {joint_commands[:, 0, :].max():.3f}]")
#     print(f"Velocity commands range: [{joint_commands[:, 1, :].min():.3f}, {joint_commands[:, 1, :].max():.3f}]")
#     print(f"Torque commands range: [{joint_commands[:, 2, :].min():.3f}, {joint_commands[:, 2, :].max():.3f}]")
#
#     # Test with different sequence lengths
#     backbone_output_variable = torch.randn(batch_size, 32, input_hidden)  # Different sequence length
#     joint_commands_variable = output_decoder(backbone_output_variable)
#     print(f"Variable sequence input shape: {backbone_output_variable.shape}")  # [4, 32, 512]
#     print(f"Variable sequence output shape: {joint_commands_variable.shape}")  # [4, 3, 12]


# class ESkinEncoding(BaseInputEncoding):
#     """
#     E-Skin Sensor ID Encoding module for spatiotemporal consistency fusion
#     in material point-based robotic systems.
#
#     This module encodes integer sensor IDs from electronic skin systems into
#     dense vector representations, specifically designed to prepare sensor
#     identification data for spatiotemporal consistency fusion across material
#     point collections. The embedding captures both spatial relationships
#     between sensor locations and temporal coherence patterns for unified
#     representation learning in dynamic robotic environments.
#     """
#
#     def __init__(self, num_sensors: int, hidden_dim: int, padding_idx = None,
#                  max_norm = None, **kwargs):
#         """
#         Initialize the E-Skin Sensor ID Encoding module for spatiotemporal fusion.
#
#         Args:
#             num_sensors: Total number of unique sensor IDs in the electronic skin system
#             hidden_dim: Dimensionality of output embedding vectors for fusion preparation
#             padding_idx: Optional index for padding sensors (excluded from gradient updates)
#             max_norm: Optional maximum norm for embedding vectors to ensure numerical stability
#             **kwargs: Additional arguments for base class initialization
#         """
#         super(ESkinEncoding, self).__init__(
#             hidden_dim=hidden_dim,
#             encoding_type='e_skin_spatiotemporal',
#             **kwargs
#         )
#
#         self.num_sensors = num_sensors
#         self.hidden_dim = hidden_dim
#
#         # Core embedding layer for sensor ID to feature vector transformation
#         self.embedding = nn.Embedding(
#             num_embeddings=num_sensors,
#             embedding_dim=hidden_dim,
#             padding_idx=padding_idx,
#             max_norm=max_norm
#         )
#
#         # Spatiotemporal consistency enhancement layers
#         self.consistency_projection = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#
#         # Layer normalization optimized for spatiotemporal fusion
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#
#
#     def forward(self, sensor_ids: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for e-skin sensor ID encoding with spatiotemporal fusion preparation.
#
#         Args:
#             sensor_ids: Electronic skin sensor ID tensor for material point fusion
#                 Shape: [batch_size, n, 1]
#                 where:
#                     - batch_size: Number of material point collections in batch
#                     - n: Number of sensor points per collection (material points)
#                     - 1: Integer sensor ID for spatiotemporal identification
#
#         Returns:
#             encoded_output: Sensor embeddings prepared for spatiotemporal consistency fusion
#                 Shape: [batch_size, n, hidden_dim]
#
#         Processing Steps:
#         1. Embedding lookup for sensor ID to feature space transformation
#         2. Spatiotemporal consistency projection for fusion preparation
#         3. Adaptive scaling and normalization for stable fusion integration
#         4. Output formatting for material point collection processing
#
#         Note:
#             - Designed specifically for material point method (MPM) integration
#             - Embeddings capture both spatial sensor layout and temporal coherence
#             - Output features are optimized for cross-modal spatiotemporal fusion
#             - Supports variable numbers of material points (n) per collection
#         """
#         # Prepare sensor IDs for embedding: [batch_size, n, 1] → [batch_size, n]
#         sensor_indices = sensor_ids.squeeze(-1).long()
#
#         # Core embedding: Integer IDs → Dense spatiotemporal features
#         embedded = self.embedding(sensor_indices)  # [batch_size, n, hidden_dim]
#
#         # Enhance features for spatiotemporal consistency fusion
#         consistency_features = self.consistency_projection(embedded)
#
#         # Apply fusion-optimized normalization
#         encoded_output = self.layer_norm(consistency_features)
#
#         return encoded_output
#
#
#
# # Example usage for spatiotemporal fusion preparation
# if __name__ == "__main__":
#     # Test the spatiotemporal ESkinEncoding module
#     batch_size = 4
#     num_material_points = 16  # Material points in collection
#     num_sensors = 256  # Unique sensors in electronic skin
#     hidden_dim = 128  # Fusion-ready feature dimension
#
#     # Create test data representing material point sensor IDs
#     id_tensor = torch.randn(batch_size, 8)  # System identity
#     sensor_ids = torch.randint(0, num_sensors, (batch_size, num_material_points, 1))
#
#     # Initialize spatiotemporal encoder
#     spatiotemporal_encoder = ESkinEncoding(
#         num_sensors=num_sensors,
#         hidden_dim=hidden_dim,
#         padding_idx=0,
#         max_norm=2.0
#     )
#
#     # Forward pass for fusion preparation
#     fusion_features = spatiotemporal_encoder(sensor_ids)
#
#     print("Spatiotemporal ESkinEncoding Test Results:")
#     print(f"Input sensor IDs shape: {sensor_ids.shape}")  # [4, 16, 1]
#     print(f"Output fusion features shape: {fusion_features.shape}")  # [4, 16, 128]


# class JointEncoding(BaseInputEncoding):
#     """
#     Joint-wise Input Encoding module for processing joint-specific data.
#
#     This module applies separate linear transformations to each joint in the input data,
#     allowing each joint to have its own learned encoding parameters. This is particularly
#     useful for robotic systems where different joints may have different characteristics
#     and behaviors that require specialized encoding.
#
#     The encoding process maintains the joint-wise structure while projecting each joint's
#     scalar value into a higher-dimensional hidden space for subsequent processing.
#     """
#
#     def __init__(self, joint_dim: int, hidden_dim: int, **kwargs):
#         """
#         Initialize the Joint Encoding module.
#
#         Args:
#             joint_dim: Number of joints to encode (fixed at 12 for this implementation)
#             hidden_dim: Hidden dimension size for the encoding
#             **kwargs: Additional arguments for base class
#         """
#         super(JointEncoding, self).__init__(
#             hidden_dim=hidden_dim,
#             encoding_type='joint_wise',
#             **kwargs
#         )
#
#         self.joint_dim = joint_dim
#
#         # Create separate linear encoders for each joint
#         # Each joint gets its own nn.Linear(1, hidden_dim) transformation
#         self.joint_encoders = nn.ModuleList([
#             nn.Linear(1, hidden_dim) for _ in range(joint_dim)
#         ])
#
#         # Optional: Layer normalization for stabilized training
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#
#         # Optional: Activation function
#         self.activation = nn.GELU()
#
#     def forward(self, joint_data: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for joint-wise encoding.
#
#         Args:
#             joint_data: Joint input data tensor containing scalar values for each joint
#                 Shape: [batch_size, joint_dim, 1]
#
#         Returns:
#             encoded_output: Joint-wise encoded features
#                 Shape: [batch_size, joint_dim, hidden_dim]
#
#         Processing Steps:
#         1. Apply joint-specific linear transformations
#         2. Apply activation function (optional)
#         3. Apply layer normalization (optional)
#
#         Note:
#             - Each of the 12 joints is processed by its own dedicated linear layer
#             - The identity tensor (id_tensor) is part of the interface but may not be used
#               in all implementations
#             - Output maintains the same joint dimension (12) but expands feature dimension
#         """
#
#         batch_size, joint_dim, _ = joint_data.shape
#
#         # Ensure we have the expected number of joints
#         if joint_dim != self.joint_dim:
#             raise ValueError(f"Expected {self.joint_dim} joints, but got {joint_dim}")
#
#         # Process each joint with its dedicated encoder
#         encoded_joints = []
#         for joint_idx in range(joint_dim):
#             # Extract data for current joint: [batch_size, 1]
#             joint_specific_data = joint_data[:, joint_idx, :]
#
#             # Apply joint-specific linear transformation: [batch_size, 1] → [batch_size, hidden_dim]
#             joint_encoded = self.joint_encoders[joint_idx](joint_specific_data)
#             encoded_joints.append(joint_encoded)
#
#         # Stack encoded joints along joint dimension
#         # Result shape: [batch_size, joint_dim, hidden_dim]
#         encoded_output = torch.stack(encoded_joints, dim=1)
#
#         # Apply activation function
#         encoded_output = self.activation(encoded_output)
#
#         # Apply layer normalization for training stability
#         encoded_output = self.layer_norm(encoded_output)
#
#         return encoded_output
#
# if __name__ == "__main__":
#     # Test the spatiotemporal ESkinEncoding module
#     batch_size = 4
#     num_material_points = 16  # Material points in collection
#     joint_dim = 12  # Unique sensors in electronic skin
#     hidden_dim = 128  # Fusion-ready feature dimension
#
#     # Create test data representing material point sensor IDs
#     id_tensor = torch.randn(batch_size, 8)  # System identity
#     sensor_ids = torch.randn(batch_size, joint_dim, 1)
#
#     # Initialize spatiotemporal encoder
#     spatiotemporal_encoder = JointEncoding(
#         joint_dim=joint_dim,
#         hidden_dim=hidden_dim,
#     )
#
#     # Forward pass for fusion preparation
#     fusion_features = spatiotemporal_encoder(sensor_ids)
#
#     print("Spatiotemporal ESkinEncoding Test Results:")
#     print(f"Input sensor IDs shape: {sensor_ids.shape}")  # [4, 16, 1]
#     print(f"Output fusion features shape: {fusion_features.shape}")  # [4, 16, 128]


# class MultiModalFusion(nn.Module):
#     """
#     Multi-modal Fusion module for cross-attention based integration of material point
#     and joint state encodings in robotic systems.
#
#     This module performs hierarchical cross-modal fusion between material point
#     encodings (from ESkinEncoding) and joint state encodings (position and velocity
#     from JointEncoding) using a 4-layer stack of MultiModalFusionBlock. Each layer
#     refines the material point representations by attending to joint state information,
#     enabling spatiotemporal consistency in material point system modeling.
#     """
#
#     def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
#                  num_layers: int = 4, **kwargs):
#         """
#         Initialize the Multi-modal Fusion module.
#
#         Args:
#             hidden_dim: Hidden dimension size for all input and output features
#             num_heads: Number of attention heads in cross-attention layers
#             dropout: Dropout rate for regularization
#             num_layers: Number of MultiModalFusionBlock layers (default: 4)
#             **kwargs: Additional keyword arguments
#         """
#         super(MultiModalFusion, self).__init__(**kwargs)
#
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#
#         # Create a stack of multi-modal fusion blocks
#         self.fusion_layers = nn.ModuleList([
#             MultiModalFusionBlock(
#                 num_query=hidden_dim,
#                 num_key=hidden_dim,
#                 num_value=hidden_dim,
#                 hidden_dim=hidden_dim,
#                 num_heads=num_heads,
#                 norm_shape=hidden_dim,
#                 ffn_num_inputs=hidden_dim,
#                 ffn_hidden_dim=hidden_dim * 4,
#                 ffn_num_outputs=hidden_dim,
#                 dropout=dropout
#             ) for _ in range(num_layers)
#         ])
#
#     def forward(self, material_point_encoding: torch.Tensor,
#                 joint_position_encoding: torch.Tensor,
#                 joint_velocity_encoding: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for multi-modal fusion of material point and joint encodings.
#
#         Args:
#             material_point_encoding: Encoded material point features from ESkinEncoding
#                 Shape: [batch_size, n, hidden_dim]
#             joint_position_encoding: Encoded joint position features from JointEncoding
#                 Shape: [batch_size, 12, hidden_dim]
#             joint_velocity_encoding: Encoded joint velocity features from JointEncoding
#                 Shape: [batch_size, 12, hidden_dim]
#
#         Returns:
#             fused_material_points: Refined material point features after multi-modal fusion
#                 Shape: [batch_size, n, hidden_dim]
#
#         Processing Steps:
#         1. Initialize with original material point encodings as query
#         2. For each fusion layer:
#            - Use current material points as Q
#            - Use joint positions as K (fixed across layers)
#            - Use joint velocities as V (fixed across layers)
#            - Update material point representations
#         3. Return final fused material point features
#
#         Note:
#             - Joint position and velocity encodings remain constant across all layers
#             - Only material point representations are updated through cross-attention
#             - Supports variable numbers of material points (n) while joints are fixed at 12
#             - Designed for spatiotemporal consistency in material point system modeling
#         """
#         # Initialize with input material point encodings
#         current_material_points = material_point_encoding
#
#         # Apply multi-layer fusion with joint state information
#         for layer in self.fusion_layers:
#             # Q: current material points, K: joint positions, V: joint velocities
#             current_material_points = layer(
#                 current_material_points,
#                 joint_position_encoding,
#                 joint_velocity_encoding
#             )
#
#         return current_material_points
#
#
# # Example usage and testing
# if __name__ == "__main__":
#     # Test the MultiModalFusion module
#     batch_size = 4
#     num_material_points = 32
#     hidden_dim = 256
#
#     # Create test data
#     material_point_encoding = torch.randn(batch_size, num_material_points, hidden_dim)
#     joint_position_encoding = torch.randn(batch_size, 12, hidden_dim)
#     joint_velocity_encoding = torch.randn(batch_size, 12, hidden_dim)
#
#     # Initialize multi-modal fusion module
#     fusion_module = MultiModalFusion(
#         hidden_dim=hidden_dim,
#         num_heads=8,
#         dropout=0.1,
#         num_layers=4
#     )
#
#     # Forward pass
#     fused_output = fusion_module(
#         material_point_encoding,
#         joint_position_encoding,
#         joint_velocity_encoding
#     )
#
#     print("MultiModalFusion Test Results:")
#     print(f"Input material points shape: {material_point_encoding.shape}")  # [4, 32, 256]
#     print(f"Input joint positions shape: {joint_position_encoding.shape}")  # [4, 12, 256]
#     print(f"Input joint velocities shape: {joint_velocity_encoding.shape}")  # [4, 12, 256]
#     print(f"Output fused features shape: {fused_output.shape}")  # [4, 32, 256]
#     print(f"Number of fusion layers: {len(fusion_module.fusion_layers)}")  # 4


# class EncoderBackbone(nn.Module):
#     """
#     Self-Attention Backbone Encoder for spatial morphology prediction of material point collections.
#
#     This module performs self-attention based encoding on fused material point features
#     to predict the spatial morphology of the robotic body and prepare representations
#     for the decoder. It consists of multiple self-attention layers that progressively
#     refine material point representations through intra-point relationships and
#     global context modeling.
#     """
#
#     def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
#                  num_layers: int = 6, **kwargs):
#         """
#         Initialize the Self-Attention Backbone Encoder.
#
#         Args:
#             hidden_dim: Hidden dimension size for input and output features
#             num_heads: Number of attention heads in self-attention layers
#             dropout: Dropout rate for regularization
#             num_layers: Number of SelfAttentionBlock layers in the backbone
#             **kwargs: Additional keyword arguments
#         """
#         super(EncoderBackbone, self).__init__(**kwargs)
#
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#
#         # Create a stack of self-attention blocks
#         self.self_attention_layers = nn.ModuleList([
#             SelfAttentionBlock(
#                 num_query=hidden_dim,
#                 num_key=hidden_dim,
#                 num_value=hidden_dim,
#                 hidden_dim=hidden_dim,
#                 num_heads=num_heads,
#                 norm_shape=hidden_dim,
#                 ffn_num_inputs=hidden_dim,
#                 ffn_hidden_dim=hidden_dim * 4,
#                 ffn_num_outputs=hidden_dim,
#                 dropout=dropout
#             ) for _ in range(num_layers)
#         ])
#
#     def forward(self, fused_material_points: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for self-attention backbone encoding.
#
#         Args:
#             fused_material_points: Fused material point features from MultiModalFusion
#                 Shape: [batch_size, n, hidden_dim]
#                 where:
#                     - batch_size: Number of material point collections in batch
#                     - n: Number of material points in each collection
#                     - hidden_dim: Feature dimension after multi-modal fusion
#
#         Returns:
#             encoded_material_points: Self-attention encoded material point features
#                 Shape: [batch_size, n, hidden_dim]
#
#         Processing Steps:
#         1. Initialize with fused material point features as input to first layer
#         2. For each self-attention layer:
#            - Use current material points as Q, K, V (self-attention)
#            - Apply self-attention with residual connections and FFN
#            - Update material point representations
#         3. Return final encoded material point features for decoder input
#
#         Note:
#             - Each layer performs pure self-attention (Q=K=V=material_points)
#             - Material point representations are progressively refined through layers
#             - Output maintains the same dimensionality for seamless decoder integration
#             - Supports variable numbers of material points (n) while preserving relationships
#         """
#         # Initialize with fused material point features
#         current_material_points = fused_material_points
#
#         # Apply multi-layer self-attention encoding
#         for layer in self.self_attention_layers:
#             # Self-attention: Q, K, V all from current material points
#             current_material_points = layer(
#                 current_material_points,  # queries
#                 current_material_points,  # keys
#                 current_material_points  # values
#             )
#
#         return current_material_points
#
#
# # Example usage and testing
# if __name__ == "__main__":
#     # Test the EncoderBackbone module
#     batch_size = 4
#     num_material_points = 32
#     hidden_dim = 256
#
#     # Create test data (output from MultiModalFusion)
#     fused_material_points = torch.randn(batch_size, num_material_points, hidden_dim)
#
#     # Initialize encoder backbone
#     encoder_backbone = EncoderBackbone(
#         hidden_dim=hidden_dim,
#         num_heads=8,
#         dropout=0.1,
#         num_layers=6
#     )
#
#     # Forward pass
#     encoded_output = encoder_backbone(fused_material_points)
#
#     print("EncoderBackbone Test Results:")
#     print(f"Input fused material points shape: {fused_material_points.shape}")  # [4, 32, 256]
#     print(f"Output encoded features shape: {encoded_output.shape}")  # [4, 32, 256]
#     print(f"Number of self-attention layers: {len(encoder_backbone.self_attention_layers)}")  # 6
#     print(f"Feature statistics - Mean: {encoded_output.mean():.3f}, Std: {encoded_output.std():.3f}")

# class DecoderFusion(nn.Module):
#     """
#     Decoder Fusion module for cross-attention based integration of spatial encodings
#     in the decoder pathway.
#
#     This module performs hierarchical cross-modal fusion between spatial position encodings
#     and spatial velocity/force encodings using a 4-layer stack of MultiModalFusionBlock.
#     Each layer refines the spatial position representations by attending to velocity and
#     force information, enabling dynamic motion modeling and force-aware decoding.
#     """
#
#     def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
#                  num_layers: int = 4, **kwargs):
#         """
#         Initialize the Decoder Fusion module.
#
#         Args:
#             hidden_dim: Hidden dimension size for all input and output features
#             num_heads: Number of attention heads in cross-attention layers
#             dropout: Dropout rate for regularization
#             num_layers: Number of MultiModalFusionBlock layers (default: 4)
#             **kwargs: Additional keyword arguments
#         """
#         super(DecoderFusion, self).__init__(**kwargs)
#
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#
#         # Create a stack of multi-modal fusion blocks
#         self.fusion_layers = nn.ModuleList([
#             MultiModalFusionBlock(
#                 num_query=hidden_dim,
#                 num_key=hidden_dim,
#                 num_value=hidden_dim,
#                 hidden_dim=hidden_dim,
#                 num_heads=num_heads,
#                 norm_shape=hidden_dim,
#                 ffn_num_inputs=hidden_dim,
#                 ffn_hidden_dim=hidden_dim * 4,
#                 ffn_num_outputs=hidden_dim,
#                 dropout=dropout
#             ) for _ in range(num_layers)
#         ])
#
#     def forward(self, position_encoding: torch.Tensor,
#                 velocity_encoding: torch.Tensor,
#                 force_encoding: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for decoder fusion of spatial encodings.
#
#         Args:
#             position_encoding: Spatial position features from SpatialEncoding
#                 Shape: [batch_size, n, hidden_dim]
#             velocity_encoding: Spatial velocity features from SpatialEncoding
#                 Shape: [batch_size, n, hidden_dim]
#             force_encoding: Spatial force features from SpatialEncoding
#                 Shape: [batch_size, n, hidden_dim]
#
#         Returns:
#             fused_position: Refined position features after multi-modal fusion
#                 Shape: [batch_size, n, hidden_dim]
#
#         Processing Steps:
#         1. Initialize with original position encodings as query
#         2. For each fusion layer:
#            - Use current position features as Q
#            - Use velocity encodings as K (fixed across layers)
#            - Use force encodings as V (fixed across layers)
#            - Update position representations
#         3. Return final fused position features
#
#         Note:
#             - Velocity and force encodings remain constant across all layers
#             - Only position representations are updated through cross-attention
#             - Supports variable spatial points (n) while maintaining feature consistency
#             - Designed for dynamic motion modeling with force awareness
#         """
#         # Initialize with input position encodings
#         current_position = position_encoding
#
#         # Apply multi-layer fusion with velocity and force information
#         for layer in self.fusion_layers:
#             # Q: current position, K: velocity encodings, V: force encodings
#             current_position = layer(
#                 current_position,  # queries (evolving)
#                 velocity_encoding,  # keys (fixed)
#                 force_encoding  # values (fixed)
#             )
#
#         return current_position
#
#
# # Example usage and testing
# if __name__ == "__main__":
#     # Test the DecoderFusion module
#     batch_size = 4
#     num_points = 16
#     hidden_dim = 256
#
#     # Create test data from SpatialEncoding instances
#     position_encoding = torch.randn(batch_size, num_points, hidden_dim)
#     velocity_encoding = torch.randn(batch_size, num_points, hidden_dim)
#     force_encoding = torch.randn(batch_size, num_points, hidden_dim)
#
#     # Initialize decoder fusion module
#     decoder_fusion = DecoderFusion(
#         hidden_dim=hidden_dim,
#         num_heads=8,
#         dropout=0.1,
#         num_layers=4
#     )
#
#     # Forward pass
#     fused_output = decoder_fusion(
#         position_encoding,
#         velocity_encoding,
#         force_encoding
#     )
#
#     print("DecoderFusion Test Results:")
#     print(f"Input position encoding shape: {position_encoding.shape}")  # [4, 16, 256]
#     print(f"Input velocity encoding shape: {velocity_encoding.shape}")  # [4, 16, 256]
#     print(f"Input force encoding shape: {force_encoding.shape}")  # [4, 16, 256]
#     print(f"Output fused features shape: {fused_output.shape}")  # [4, 16, 256]
#     print(f"Number of fusion layers: {len(decoder_fusion.fusion_layers)}")  # 4


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


# Example usage and testing
if __name__ == "__main__":
    # Test the DecoderBlock module
    batch_size = 4
    num_points = 16
    hidden_dim = 256

    # Create test data
    fused_target_states = torch.randn(batch_size, num_points, hidden_dim)
    encoder_output = torch.randn(batch_size, num_points, hidden_dim)

    # Initialize decoder block
    decoder_block = DecoderBlock(
        hidden_dim=hidden_dim,
        num_heads=8,
        dropout=0.1,
        num_layers=6
    )

    # Forward pass
    decoded_output = decoder_block(fused_target_states, encoder_output)

    print("DecoderBlock Test Results:")
    print(f"Input target states shape: {fused_target_states.shape}")  # [4, 16, 256]
    print(f"Input encoder output shape: {encoder_output.shape}")  # [4, 16, 256]
    print(f"Output decoded features shape: {decoded_output.shape}")  # [4, 16, 256]
    print(f"Number of fusion layers: {len(decoder_block.fusion_layers)}")  # 6
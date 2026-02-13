import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K

# ====================================================================
# A. Custom Functions (PyTorch to TensorFlow)
# ====================================================================

def get_activation_tf(activation):
    """PyTorch의 get_activation을 Keras/TF 활성화 함수로 변환"""
    if activation == "relu":
        return layers.ReLU()
    elif activation == "gelu":
        return layers.Activation('gelu') 
    elif activation == "leaky":
        return layers.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")

def shift_with_padding_tf(x, shift, axis):
    """
    shift_with_padding 함수를 TensorFlow 텐서 연산으로 구현.
    PyTorch의 torch.narrow와 torch.cat을 tf.slice와 tf.pad로 대체.
    """
    if shift == 0:
        return x
        
    x_shape = tf.shape(x)
    size = x_shape[axis]
    abs_shift = tf.abs(shift)
    
    # 텐서의 차원 수
    rank = tf.rank(x)
    
    if shift > 0:
        # 양수 shift (오른쪽 이동): 왼쪽에 0을 추가 ([0, 0, 0, keep, keep])
        keep_size = size - abs_shift
        
        # 1. 유지할 부분 (keep) 슬라이싱
        begin = tf.zeros(rank, dtype=tf.int32)
        size_slice = tf.tensor_scatter_nd_update(x_shape, [[axis]], [keep_size])
        keep = tf.slice(x, begin, size_slice)
        
        # 2. 패딩을 위한 설정 (axis=dim, 왼쪽에 abs_shift만큼)
        paddings = tf.zeros((rank, 2), dtype=tf.int32)
        paddings = tf.tensor_scatter_nd_update(paddings, [[axis, 0]], [abs_shift])
        
        shifted = tf.pad(keep, paddings, "CONSTANT")
    else:
        # 음수 shift (왼쪽 이동): 오른쪽에 0을 추가 ([keep, keep, 0, 0, 0])
        keep_size = size - abs_shift
        
        # 1. 유지할 부분 (keep) 슬라이싱
        begin = tf.tensor_scatter_nd_update(tf.zeros(rank, dtype=tf.int32), [[axis]], [abs_shift])
        size_slice = tf.tensor_scatter_nd_update(x_shape, [[axis]], [keep_size])
        keep = tf.slice(x, begin, size_slice)
        
        # 2. 패딩을 위한 설정 (axis=dim, 오른쪽에 abs_shift만큼)
        paddings = tf.zeros((rank, 2), dtype=tf.int32)
        paddings = tf.tensor_scatter_nd_update(paddings, [[axis, 1]], [abs_shift])
        
        shifted = tf.pad(keep, paddings, "CONSTANT")

    return shifted

def channel_shift_tf(x, shift, shift_size):
    """
    PyTorch의 channel_shift 함수를 TensorFlow로 구현.
    x는 PyTorch와 동일하게 (B, C, N) 형태를 가정.
    Shift는 N 차원 (axis=2)에 적용됨.
    """
    # x는 (B, C, N)
    x_chunk = tf.split(x, shift_size, axis=1) # Channel 차원 (axis=1)으로 나누기
    shifted_chunks = []

    for chunk, sh in zip(x_chunk, shift):
        # Shift는 N 차원 (axis=2)에 적용됨
        shifted = shift_with_padding_tf(chunk, sh, axis=2) 
        shifted_chunks.append(shifted)
        
    # Channel 차원 (axis=1)으로 다시 합치기
    x_shifted = tf.concat(shifted_chunks, axis=1) 

    return x_shifted

# ====================================================================
# B. Custom Layers (PyTorch Modules to Keras Layers)
# ====================================================================

class PositionalEmbedding(layers.Layer):
    """PyTorch PositionalEmbedding을 Keras Layer로 구현"""
    def __init__(self, d_feature, max_len, **kwargs):
        super().__init__(**kwargs)
        
        position_indices = tf.range(0, max_len, dtype=tf.float32)
        position_indices = tf.expand_dims(tf.expand_dims(position_indices, 0), 0)
        
        # (1, C, max_len) 형태로 반복 및 버퍼 등록
        self.position = tf.tile(position_indices, [1, d_feature, 1])
        # Keras에서는 add_weight(trainable=False)로 버퍼 등록 효과를 냅니다.
        self.positional_embedding = self.add_weight(
            shape=self.position.shape,
            initializer=tf.constant_initializer(self.position.numpy()),
            trainable=False,
            name='positional_embedding'
        )

    def call(self, x):
        # x는 (B, C, N)
        B, C, N = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        
        # PyTorch 로직: x + self.positional_embedding[:, :, :N]
        # Keras 텐서 슬라이싱
        pos_emb_slice = self.positional_embedding[:, :, :N]

        return x + pos_emb_slice

class MlpBlock(layers.Layer):
    """PyTorch MlpBlock을 Keras Sequential/Layer로 구현"""
    def __init__(self, in_features, out_features, activation, dropout, **kwargs):
        super().__init__(**kwargs)
        # Keras Dense는 마지막 축에 적용되므로, PyTorch Linear와 동일한 효과를 냅니다.
        self.mlp = keras.Sequential([
            layers.Dense(out_features),
            get_activation_tf(activation),
            layers.Dropout(dropout)
        ])
        
    def call(self, x):
        return self.mlp(x)

class ShiftBlock(layers.Layer):
    """PyTorch ShiftBlock을 Keras Layer로 구현"""
    def __init__(self, patch_dim, num_patches, shift=[-1, 0, 1], shift_size=3, dropout=0.1, act='relu', **kwargs):
        super().__init__(**kwargs)
        self.shift = shift
        self.shift_size = shift_size
        self.act = act

        # Channel Mixer S: (B, N, C) 입력 가정 -> LayerNorm(-1)
        self.channel_mixer_S = keras.Sequential([
            layers.LayerNormalization(axis=-1),
            MlpBlock(patch_dim, patch_dim, self.act, dropout)
        ])

        # Channel Projection: (B, N, C) 입력 가정 (N에 대해 Dense 적용)
        self.channel_projection = keras.Sequential([
            MlpBlock(num_patches, num_patches * 2, self.act, dropout),
            layers.Dense(num_patches)
        ])

        # SE Block Squeeze/Excitation
        self.squeeze = layers.GlobalAveragePooling1D(data_format='channels_first') # (B, C, N) -> (B, C)
        self.excitation = keras.Sequential([
            layers.Dense(num_patches // 8),
            get_activation_tf(self.act),
            layers.Dense(num_patches),
            layers.Activation('sigmoid')
        ])

        # Channel Mixer F
        self.channel_mixer_F = keras.Sequential([
            MlpBlock(patch_dim, patch_dim * 2, self.act, dropout),
            layers.Dense(patch_dim)
        ])

    def call(self, x):
        # x는 (B, N, C)를 가정 (Keras 표준)
        res = x
        x = self.channel_mixer_S(x)

        # 1. Channel Shift (PyTorch와 동일한 로직을 위해 (B, C, N)으로 변환 후 연산)
        x_shift = tf.transpose(x, [0, 2, 1]) # (B, C, N)
        x_shift = channel_shift_tf(x_shift, shift=self.shift, shift_size=self.shift_size)

        # 2. Channel Projection (N 차원=axis=2에 대해 연산)
        x_shift = tf.transpose(x_shift, [0, 2, 1]) # (B, N, C)로 복귀
        x_shift = self.channel_projection(x_shift) # (B, N, C)

        # 3. SE Block (Squeeze/Excitation)
        se_input = tf.transpose(x_shift, [0, 2, 1]) # (B, C, N)
        # Squeeze: (B, C, N) -> Global Average Pooling over N -> (B, C)
        se = layers.GlobalAveragePooling1D(data_format='channels_first')(se_input)
        
        ex = self.excitation(se) # (B, C)
        ex = tf.expand_dims(ex, axis=1) # (B, 1, C)

        z = x_shift * ex # 브로드캐스팅 (B, N, C) * (B, 1, C)

        # 4. Final Channel Mixer (F) 및 Skip Connection
        z = self.channel_mixer_F(z) + res

        return z

class Downsample(layers.Layer):
    """PyTorch Downsample을 Keras Layer로 구현"""
    def __init__(self, in_channels, **kwargs):
        super().__init__(**kwargs)
        # PyTorch 코드는 (B, C, N) -> (B, N, C) -> Norm -> (B, C, N) -> Conv1d(kernel=2, stride=2)
        self.norm = layers.LayerNormalization(axis=-1)
        # Keras Conv1D (channels_first) 사용으로 (B, C, N) 형태 유지
        self.reduction = layers.Conv1D(filters=in_channels,
                                       kernel_size=2,
                                       strides=2,
                                       padding='valid',
                                       data_format='channels_first')

    def call(self, x):
        # x는 (B, C, N)을 가정
        
        x = tf.transpose(x, [0, 2, 1]) # (B, N, C)
        x = self.norm(x)
        x = tf.transpose(x, [0, 2, 1]) # (B, C, N)
        
        x = self.reduction(x) # (B, C, N/2)
        
        return x

class BasicLayer(layers.Layer):
    """PyTorch BasicLayer를 Keras Layer로 구현"""
    def __init__(self, patch_dim, num_patches, num_layers, shift=[-1, 0, 1], shift_size=3, dropout=0.1, downsample=False, act='relu', **kwargs):
        super().__init__(**kwargs)
        
        self.shift_blocks = []
        self.token_mixers = []
        
        # 레이어 반복 정의
        for _ in range(num_layers):
            # Shift Block: (B, N, C) -> ShiftBlock -> (B, N, C)
            self.shift_blocks.append(
                ShiftBlock(patch_dim, num_patches, shift, shift_size, dropout, act)
            )
            # Token Mixer: (B, C, N) -> (B, N, C) -> Dense -> (B, N, C)
            self.token_mixers.append(
                keras.Sequential([
                    # PyTorch 코드는 nn.LayerNorm(num_patches) 사용 -> (B, C, N)에서 N 차원에 대한 LN
                    # Keras는 LayerNorm(axis=-1)이 기본이므로, 여기서는 N 차원에 적용하기 위해 복잡한 트랜스포즈가 필요.
                    # PyTorch 코드를 직접 모방하기 위해 (B, C, N) -> (B, C) -> Norm -> (B, C) 로 변환하는 것이 정확한지는 불분명하므로,
                    # 가장 유사한 Dense(Token Mixer) 로직에 맞게 (B, N, C)에서 N 차원을 특징으로 간주하고 LayerNorm을 마지막 차원 C에 적용.
                    # PyTorch 코드의 TokenMixer는 (B, C, N)에서 N에 대한 Mixer이므로 (B, C)에 Dense를 적용해야 함.
                    
                    # Token Mixer: (B, C, N)을 입력받아 N차원(Sequence)에 대한 Mixer를 수행해야 함.
                    # Keras Dense는 마지막 차원에 적용되므로, (B, C, N)을 (B, N, C)로 변환하여 N에 대해 Dense를 적용합니다.
                    MlpBlock(num_patches, num_patches * 2, act, dropout),
                    layers.Dense(num_patches)
                ])
            )
            
        self.downsample = Downsample(patch_dim) if downsample else None

    def call(self, x):
        # x는 (B, C, N)을 가정
        results = []
        current_x = x
        
        for shift_blk, token_blk in zip(self.shift_blocks, self.token_mixers):
            
            # 1. Shift Block (Channel Mixer)
            # 입력 형태 (B, C, N) -> (B, N, C)로 변환하여 ShiftBlock에 전달
            x_shifted = tf.transpose(current_x, [0, 2, 1]) # (B, N, C)
            x_shifted = shift_blk(x_shifted) # (B, N, C)
            
            # 2. Token Mixer
            # Shift Block 출력 (B, N, C)를 다시 (B, C, N)으로 변환
            x_shifted_permuted = tf.transpose(x_shifted, [0, 2, 1]) # (B, C, N)
            
            # Token Mixer는 N 차원 (Sequence)에 대한 Mixer를 수행해야 함.
            # (B, C, N)을 (B, C, N) -> (B*C, N) 형태로 변환하여 N에 대한 Dense를 적용해야 함.
            # 가장 간단한 방법: (B, C, N) -> (B, N, C)로 변환하여 N 차원에 Dense를 적용
            x_token_input = tf.transpose(x_shifted_permuted, [0, 2, 1]) # (B, N, C)
            x_token_mixed = token_blk(x_token_input) # (B, N, C)
            
            # 결과 복귀: (B, N, C) -> (B, C, N)
            x_token_mixed = tf.transpose(x_token_mixed, [0, 2, 1])
            
            # Skip Connection: PyTorch 로직 x = token(x) + x
            current_x = x_token_mixed + x_shifted_permuted
            
            results.append(current_x)
            
        if self.downsample is not None:
            current_x = self.downsample(current_x)

        return current_x, results


class MultiscaleMixer(Model):
    """
    Multi-Scale Patch Shift Mixer
    """
    def __init__(self, 
                 in_channels=3, 
                 patch_dim=128, 
                 dropout=0.1, 
                 num_layers=[2, 2], 
                 patches=[(224, 2), (224, 4)], 
                 stride=[(224, 2), (224, 4)], 
                 shift_size=4, shift=[3,-2,2,-3], 
                 num_patches=[112, 56], 
                 act='relu', 
                 **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.patches = patches
        self.stride = stride
        self.shift_size = shift_size
        self.shift = shift
        self.act = act
        self.num_patches = num_patches
        
        # Patch Embedding: Conv2D는 Keras 표준 (B, H, W, C)를 입력으로 받음. 
        self.patch_embedding = [
            layers.Conv2D(patch_dim, kernel_size=p, strides=s, padding='valid') 
            for p, s in zip(patches, stride)
        ]
        
        # Positional Embedding: 커스텀 구현 사용
        self.positional_embedding = [
            PositionalEmbedding(d_feature=patch_dim, max_len=x)
            for x in num_patches
        ]
        
        # Basic Layers
        self.blocks = []
        for p_idx, num_p in enumerate(num_patches):
            layer_list = []
            for idx, num_l in enumerate(num_layers):
                is_downsample = False if idx == len(num_layers) - 1 else True
                # num_patches 계산 로직은 PyTorch와 동일하게 유지
                current_num_patches = num_p // (2**idx)
                layer_list.append(
                    BasicLayer(
                        patch_dim=patch_dim,
                        num_patches=current_num_patches,
                        num_layers=num_l,
                        shift=shift,
                        shift_size=shift_size,
                        dropout=dropout,
                        downsample=is_downsample,
                        act=act
                    )
                )
            self.blocks.append(layer_list)

        # Head
        self.head = keras.Sequential([
            layers.LayerNormalization(axis=-1), # (B, C) 형태에서 C에 대해 Norm
            layers.Dense(patch_dim // 2, activation='relu'),
            layers.Dense(6, activation='softmax')
        ])

    def call(self, x):
        # x는 (B, H, W, C)를 가정 (Keras 표준)
        Mixer_output = []
        
        for p_idx in range(len(self.patches)):
            # 1. Patch Embedding (Conv2D)
            z = self.patch_embedding[p_idx](x) # (B, H', W', C')
            
            # 2. Flatten(2) 및 Permute: (B, H', W', C') -> (B, N, C) -> (B, C, N)
            z_shape = tf.shape(z)
            N = z_shape[1] * z_shape[2] # H' * W'
            C = z_shape[3] # C' = patch_dim
            
            # (B, H', W', C) -> (B, N, C)
            z = tf.reshape(z, (z_shape[0], N, C)) 
            # (B, N, C) -> (B, C, N) (PyTorch의 flatten(2) 결과 형태)
            z = tf.transpose(z, [0, 2, 1]) 

            # 3. Positional Embedding (입력 형태: B, C, N)
            z = self.positional_embedding[p_idx](z)

            # 4. Basic Layers
            current_z = z
            
            for blk in self.blocks[p_idx]:
                current_z, _ = blk(current_z) # BasicLayer는 (B, C, N)을 반환
                
            Mixer_output.append(current_z) # (B, C, N)
        
        # 5. Concatenate Multi-Scale Patch
        z = tf.concat(Mixer_output, axis=2) # (B, C, N1+N2+...)

        # 6. GAP (Global Average Pooling)
        # PyTorch: torch.mean(z, dim=2) -> (B, C)
        x = tf.reduce_mean(z, axis=2) 

        # 7. Head
        logit = self.head(x)
        
        # Keras Model은 일반적으로 단일 출력을 반환하므로 logit만 반환
        return logit
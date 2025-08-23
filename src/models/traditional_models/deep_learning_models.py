"""
A2AI (Advanced Financial Analysis AI) - Deep Learning Models
企業ライフサイクル全体を考慮した深層学習による財務諸表分析

主要機能:
1. 企業存続予測モデル (LSTM-based Survival Prediction)
2. 財務指標予測モデル (Multi-task Neural Network)
3. 市場シェア変動予測モデル (Transformer-based)
4. 企業ライフサイクル分類モデル (CNN+LSTM Hybrid)
5. 要因項目影響度分析モデル (Attention-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """深層学習モデル設定クラス"""
    input_dim: int = 23  # 拡張要因項目数（従来20項目 + 企業年齢等3項目）
    output_dim: int = 9   # 拡張評価項目数（従来6項目 + 新規3項目）
    hidden_dims: List[int] = None
    sequence_length: int = 40  # 40年分の時系列データ
    num_heads: int = 8
    num_layers: int = 6
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 256, 512, 256, 128]


class SurvivalPredictionLSTM(nn.Module):
    """
    企業存続予測LSTM模型
    企業の消滅・存続を時系列財務データから予測
    """
    
    def __init__(self, config: ModelConfig):
        super(SurvivalPredictionLSTM, self).__init__()
        self.config = config
        
        # LSTM层用于时序特征提取
        self.lstm1 = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dims[0],
            num_layers=2,
            batch_first=True,
            dropout=config.dropout_rate,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=config.hidden_dims[0] * 2,  # bidirectional
            hidden_size=config.hidden_dims[1] // 2,
            num_layers=1,
            batch_first=True,
            dropout=config.dropout_rate,
            bidirectional=True
        )
        
        # Attention层
        self.attention = MultiHeadAttention(
            d_model=config.hidden_dims[1],
            num_heads=config.num_heads,
            dropout=config.dropout_rate
        )
        
        # 全连接层
        self.fc_layers = nn.ModuleList()
        dims = [config.hidden_dims[1]] + config.hidden_dims[2:] + [1]  # 存続確率(0-1)
        
        for i in range(len(dims) - 1):
            self.fc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 最後の層以外にDropoutとReLU
                self.fc_layers.append(nn.Dropout(config.dropout_rate))
                self.fc_layers.append(nn.ReLU())
        
        # 最终输出层使用Sigmoid激活
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            Dict containing survival probability and attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM特征提取
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Attention机制
        attended_features, attention_weights = self.attention(
            lstm_out2, lstm_out2, lstm_out2
        )
        
        # 全局平均池化
        pooled_features = attended_features.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 全连接层预测
        out = pooled_features
        for layer in self.fc_layers:
            out = layer(out)
        
        survival_prob = self.sigmoid(out)
        
        return {
            'survival_probability': survival_prob,
            'attention_weights': attention_weights,
            'features': attended_features
        }


class MultiTaskFinancialPredictor(nn.Module):
    """
    多任务财务指标预测模型
    同时预测9个评价项目
    """
    
    def __init__(self, config: ModelConfig):
        super(MultiTaskFinancialPredictor, self).__init__()
        self.config = config
        self.num_tasks = 9  # 9个评价项目
        
        # 共享特征提取器
        self.shared_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
        )
        
        # 时序编码器
        self.temporal_encoder = nn.LSTM(
            input_size=config.hidden_dims[1],
            hidden_size=config.hidden_dims[2],
            num_layers=3,
            batch_first=True,
            dropout=config.dropout_rate,
            bidirectional=True
        )
        
        # 任务特定的预测头
        self.task_heads = nn.ModuleDict({
            'sales_revenue': self._create_regression_head(config.hidden_dims[2] * 2),
            'sales_growth_rate': self._create_regression_head(config.hidden_dims[2] * 2),
            'operating_margin': self._create_regression_head(config.hidden_dims[2] * 2),
            'net_profit_margin': self._create_regression_head(config.hidden_dims[2] * 2),
            'roe': self._create_regression_head(config.hidden_dims[2] * 2),
            'value_added_ratio': self._create_regression_head(config.hidden_dims[2] * 2),
            'survival_probability': self._create_classification_head(config.hidden_dims[2] * 2),
            'emergence_success_rate': self._create_regression_head(config.hidden_dims[2] * 2),
            'succession_success_degree': self._create_regression_head(config.hidden_dims[2] * 2),
        })
        
        # 交叉注意力机制用于任务间信息交互
        self.cross_attention = CrossTaskAttention(
            d_model=config.hidden_dims[2] * 2,
            num_tasks=self.num_tasks,
            num_heads=config.num_heads
        )
        
    def _create_regression_head(self, input_dim: int) -> nn.Module:
        """创建回归预测头"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(input_dim // 4, 1)
        )
    
    def _create_classification_head(self, input_dim: int) -> nn.Module:
        """创建分类预测头（用于存续概率）"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Multi-task forward pass
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            Dict containing predictions for all 9 evaluation metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # 每个时间步的特征提取
        shared_features = []
        for t in range(seq_len):
            feat = self.shared_encoder(x[:, t, :])
            shared_features.append(feat)
        
        # 时序编码
        shared_features = torch.stack(shared_features, dim=1)  # (batch_size, seq_len, hidden_dim)
        temporal_features, _ = self.temporal_encoder(shared_features)
        
        # 最后时间步的特征用于预测
        final_features = temporal_features[:, -1, :]  # (batch_size, hidden_dim*2)
        
        # 任务特定预测
        task_features = []
        predictions = {}
        
        for task_name, head in self.task_heads.items():
            task_pred = head(final_features)
            predictions[task_name] = task_pred
            task_features.append(final_features)
        
        # 交叉注意力增强
        task_features_tensor = torch.stack(task_features, dim=1)  # (batch_size, num_tasks, hidden_dim)
        enhanced_features = self.cross_attention(task_features_tensor)
        
        # 基于增强特征的最终预测
        final_predictions = {}
        for i, (task_name, head) in enumerate(self.task_heads.items()):
            enhanced_pred = head(enhanced_features[:, i, :])
            final_predictions[f'{task_name}_enhanced'] = enhanced_pred
        
        predictions.update(final_predictions)
        
        return predictions


class MarketShareTransformer(nn.Module):
    """
    基于Transformer的市场シェア变动预测模型
    考虑企业间竞争关系和市场生态系统
    """
    
    def __init__(self, config: ModelConfig):
        super(MarketShareTransformer, self).__init__()
        self.config = config
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(
            d_model=config.hidden_dims[0],
            max_len=config.sequence_length
        )
        
        # 输入投影层
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dims[0])
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dims[0],
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dims[1],
            dropout=config.dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # 市场シェア预测头
        self.market_share_head = nn.Sequential(
            nn.Linear(config.hidden_dims[0], config.hidden_dims[2]),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[2], config.hidden_dims[3]),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[3], 3),  # 高シェア、低下、失失の3分类
            nn.Softmax(dim=-1)
        )
        
        # 市场健康度预测
        self.market_health_head = nn.Sequential(
            nn.Linear(config.hidden_dims[0], config.hidden_dims[2] // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[2] // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Transformer forward pass for market share prediction
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            Dict containing market share predictions and market health scores
        """
        # 输入投影和位置编码
        x_projected = self.input_projection(x)
        x_pos_encoded = self.positional_encoding(x_projected)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(x_pos_encoded)
        
        # 使用最后时间步进行预测
        final_representation = transformer_out[:, -1, :]
        
        # 市场シェア预测
        market_share_pred = self.market_share_head(final_representation)
        
        # 市场健康度预测
        market_health = self.market_health_head(final_representation)
        
        return {
            'market_share_category': market_share_pred,
            'market_health_score': market_health,
            'transformer_features': transformer_out
        }


class LifecycleClassificationCNN(nn.Module):
    """
    企业ライフサイクル分类模型 (CNN + LSTM Hybrid)
    识别企业所处的生命周期阶段
    """
    
    def __init__(self, config: ModelConfig):
        super(LifecycleClassificationCNN, self).__init__()
        self.config = config
        
        # 1D卷积层提取局部模式
        self.conv_layers = nn.ModuleList([
            # 短期模式 (kernel_size=3)
            nn.Conv1d(config.input_dim, config.hidden_dims[0], kernel_size=3, padding=1),
            # 中期模式 (kernel_size=5)  
            nn.Conv1d(config.input_dim, config.hidden_dims[0], kernel_size=5, padding=2),
            # 长期模式 (kernel_size=7)
            nn.Conv1d(config.input_dim, config.hidden_dims[0], kernel_size=7, padding=3),
        ])
        
        # 批归一化和激活
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(config.hidden_dims[0]) for _ in range(3)
        ])
        
        # 池化层
        self.pool = nn.AdaptiveMaxPool1d(config.sequence_length // 2)
        
        # LSTM层用于序列建模
        self.lstm = nn.LSTM(
            input_size=config.hidden_dims[0] * 3,  # 3个卷积分支
            hidden_size=config.hidden_dims[1],
            num_layers=2,
            batch_first=True,
            dropout=config.dropout_rate,
            bidirectional=True
        )
        
        # 生命周期阶段分类 (5个阶段: 创业期、成长期、成熟期、衰退期、转型期)
        self.lifecycle_classifier = nn.Sequential(
            nn.Linear(config.hidden_dims[1] * 2, config.hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[2], config.hidden_dims[3]),
            nn.ReLU(), 
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[3], 5),  # 5个生命周期阶段
            nn.Softmax(dim=-1)
        )
        
        # 生命周期转换概率预测
        self.transition_predictor = nn.Sequential(
            nn.Linear(config.hidden_dims[1] * 2, config.hidden_dims[3]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[3], 25),  # 5x5转换矩阵
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        CNN-LSTM forward pass for lifecycle classification
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            Dict containing lifecycle stage predictions and transition probabilities
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 转换为卷积输入格式 (batch_size, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        
        # 多尺度卷积特征提取
        conv_features = []
        for i, (conv_layer, bn_layer) in enumerate(zip(self.conv_layers, self.batch_norms)):
            conv_out = conv_layer(x_conv)
            conv_out = bn_layer(conv_out)
            conv_out = F.relu(conv_out)
            conv_out = self.pool(conv_out)
            conv_features.append(conv_out)
        
        # 合并多尺度特征
        combined_conv = torch.cat(conv_features, dim=1)  # (batch_size, hidden_dim*3, seq_len//2)
        
        # 转换为LSTM输入格式
        combined_conv = combined_conv.transpose(1, 2)  # (batch_size, seq_len//2, hidden_dim*3)
        
        # LSTM序列建模
        lstm_out, _ = self.lstm(combined_conv)
        
        # 使用最后时间步进行分类
        final_features = lstm_out[:, -1, :]
        
        # 生命周期阶段预测
        lifecycle_pred = self.lifecycle_classifier(final_features)
        
        # 转换概率预测
        transition_pred = self.transition_predictor(final_features)
        transition_matrix = transition_pred.view(batch_size, 5, 5)
        
        return {
            'lifecycle_stage': lifecycle_pred,
            'transition_matrix': transition_matrix,
            'lstm_features': lstm_out
        }


class FactorImpactAttention(nn.Module):
    """
    基于注意力机制的要因项目影响度分析模型
    量化23个要因項目对9个評価項目的影響度
    """
    
    def __init__(self, config: ModelConfig):
        super(FactorImpactAttention, self).__init__()
        self.config = config
        self.num_factors = 23  # 拡張要因項目数
        self.num_evaluations = 9  # 拡張評価項目数
        
        # 要因項目编码器
        self.factor_encoder = nn.Sequential(
            nn.Linear(1, config.hidden_dims[0] // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0] // 4, config.hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0] // 2, config.hidden_dims[0]),
        )
        
        # 多头注意力层
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dims[0],
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # 要因-評価項目影响力计算
        self.impact_calculator = nn.ModuleDict({
            f'eval_{i}': nn.Sequential(
                nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dims[1], self.num_factors),
                nn.Softmax(dim=-1)  # 影响力权重
            ) for i in range(self.num_evaluations)
        })
        
        # 时间依赖的影响力建模
        self.temporal_impact_lstm = nn.LSTM(
            input_size=self.num_factors,
            hidden_size=config.hidden_dims[1],
            num_layers=2,
            batch_first=True,
            dropout=config.dropout_rate
        )
        
        # 影响力变化趋势预测
        self.trend_predictor = nn.Sequential(
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[2], self.num_factors),
            nn.Tanh()  # 影响力变化率 [-1, 1]
        )
        
    def forward(self, factors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Factor impact analysis forward pass
        Args:
            factors: (batch_size, sequence_length, num_factors)
        Returns:
            Dict containing factor impact weights and trends for each evaluation metric
        """
        batch_size, seq_len, num_factors = factors.shape
        
        # 要因項目编码
        factor_encoded = []
        for t in range(seq_len):
            encoded_timestep = []
            for f in range(num_factors):
                factor_val = factors[:, t, f:f+1]  # (batch_size, 1)
                encoded_factor = self.factor_encoder(factor_val)  # (batch_size, hidden_dim)
                encoded_timestep.append(encoded_factor)
            
            timestep_features = torch.stack(encoded_timestep, dim=1)  # (batch_size, num_factors, hidden_dim)
            factor_encoded.append(timestep_features)
        
        factor_encoded = torch.stack(factor_encoded, dim=1)  # (batch_size, seq_len, num_factors, hidden_dim)
        
        # 使用最后时间步进行影响力分析
        last_timestep = factor_encoded[:, -1, :, :]  # (batch_size, num_factors, hidden_dim)
        
        # 自注意力机制
        attended_factors, attention_weights = self.multi_head_attention(
            last_timestep, last_timestep, last_timestep
        )
        
        # 全局特征聚合
        global_features = attended_factors.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 各評価項目的影响力计算
        impact_results = {}
        for eval_name, impact_net in self.impact_calculator.items():
            impact_weights = impact_net(global_features)  # (batch_size, num_factors)
            impact_results[f'{eval_name}_impact'] = impact_weights
        
        # 时序影响力建模 
        factor_importance_seq = factors.mean(dim=2)  # (batch_size, seq_len)
        factor_importance_seq = factor_importance_seq.unsqueeze(2).expand(-1, -1, num_factors)
        
        temporal_features, _ = self.temporal_impact_lstm(factor_importance_seq)
        
        # 影响力变化趋势
        trend_features = temporal_features[:, -1, :]  # 最后时间步
        impact_trends = self.trend_predictor(trend_features)
        
        impact_results.update({
            'attention_weights': attention_weights,
            'impact_trends': impact_trends,
            'temporal_features': temporal_features
        })
        
        return impact_results


# 辅助类和函数
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        # 线性变换
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)
        
        # 拼接多头输出
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.output_linear(attended)
        
        return output, attention_weights


class CrossTaskAttention(nn.Module):
    """任务间交叉注意力机制"""
    
    def __init__(self, d_model: int, num_tasks: int, num_heads: int):
        super(CrossTaskAttention, self).__init__()
        self.d_model = d_model
        self.num_tasks = num_tasks
        self.num_heads = num_heads
        
        self.task_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.task_fusion = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Cross-task attention forward pass
        Args:
            task_features: (batch_size, num_tasks, d_model)
        Returns:
            Enhanced task features: (batch_size, num_tasks, d_model)
        """
        attended_features, _ = self.task_attention(
            task_features, task_features, task_features
        )
        
        # 残差连接
        enhanced_features = task_features + attended_features
        
        # 特征融合
        fused_features = self.task_fusion(enhanced_features)
        
        return fused_features


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            Position encoded tensor: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x


class A2AIDeepLearningSystem:
    """
    A2AI深度学习系统的主控制类
    集成所有深度学习模型，提供统一的训练、推理和分析接口
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 初始化各个深度学习模型
        self.models = {
            'survival_lstm': SurvivalPredictionLSTM(config).to(self.device),
            'multitask_predictor': MultiTaskFinancialPredictor(config).to(self.device),
            'market_transformer': MarketShareTransformer(config).to(self.device),
            'lifecycle_cnn': LifecycleClassificationCNN(config).to(self.device),
            'factor_attention': FactorImpactAttention(config).to(self.device)
        }
        
        # 优化器设置
        self.optimizers = {
            name: torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=1e-5
            ) for name, model in self.models.items()
        }
        
        # 学习率调度器
        self.schedulers = {
            name: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.max_epochs
            ) for name, optimizer in self.optimizers.items()
        }
        
        # 损失函数
        self.loss_functions = {
            'survival': nn.BCELoss(),
            'regression': nn.MSELoss(),
            'classification': nn.CrossEntropyLoss(),
            'multitask': self._create_multitask_loss()
        }
        
        # 训练历史记录
        self.training_history = {
            'losses': {},
            'metrics': {},
            'learning_rates': {}
        }
        
    def _create_multitask_loss(self) -> callable:
        """创建多任务学习损失函数"""
        def multitask_loss(predictions: Dict[str, torch.Tensor], 
                            targets: Dict[str, torch.Tensor]) -> torch.Tensor:
            total_loss = 0.0
            task_weights = {
                'survival_probability': 2.0,  # 企业存续预测权重更高
                'sales_revenue': 1.0,
                'sales_growth_rate': 1.0, 
                'operating_margin': 1.5,
                'net_profit_margin': 1.5,
                'roe': 1.0,
                'value_added_ratio': 1.0,
                'emergence_success_rate': 1.5,
                'succession_success_degree': 1.0
            }
            
            for task_name, pred in predictions.items():
                if task_name in targets and task_name in task_weights:
                    target = targets[task_name]
                    if 'probability' in task_name:
                        loss = F.binary_cross_entropy(pred, target)
                    else:
                        loss = F.mse_loss(pred, target)
                    
                    total_loss += task_weights[task_name] * loss
            
            return total_loss
        
        return multitask_loss
    
    def prepare_data(self, financial_data: pd.DataFrame, 
                    target_data: pd.DataFrame) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        数据预处理和张量转换
        Args:
            financial_data: 财务数据DataFrame (企业 x 年份 x 要因項目)
            target_data: 目标数据DataFrame (企业 x 評価項目)
        Returns:
            输入张量和目标张量字典
        """
        # 数据标准化
        scaler = StandardScaler()
        
        # 处理输入数据 (150企业 x 40年 x 23要因项目)
        input_data = []
        for company in financial_data['company_id'].unique():
            company_data = financial_data[financial_data['company_id'] == company]
            
            # 确保时间序列完整性（处理企业消滅、新設等情况）
            company_tensor = self._process_company_timeline(company_data, scaler)
            input_data.append(company_tensor)
        
        input_tensor = torch.stack(input_data).float().to(self.device)
        
        # 处理目标数据
        target_tensors = {}
        evaluation_metrics = [
            'sales_revenue', 'sales_growth_rate', 'operating_margin',
            'net_profit_margin', 'roe', 'value_added_ratio',
            'survival_probability', 'emergence_success_rate', 'succession_success_degree'
        ]
        
        for metric in evaluation_metrics:
            if metric in target_data.columns:
                target_tensors[metric] = torch.tensor(
                    target_data[metric].values, dtype=torch.float32
                ).to(self.device)
        
        return input_tensor, target_tensors
    
    def _process_company_timeline(self, company_data: pd.DataFrame, 
                                    scaler: StandardScaler) -> torch.Tensor:
        """
        处理单个企业的时间序列数据
        处理企业消滅、新設等特殊情况
        """
        timeline_data = np.zeros((self.config.sequence_length, self.config.input_dim))
        
        # 获取企业存续期间
        years = sorted(company_data['year'].unique())
        start_year = min(years) if years else 1984
        end_year = max(years) if years else 2024
        
        # 填充存续期间的数据
        for i, year in enumerate(range(1984, 2024)):
            year_data = company_data[company_data['year'] == year]
            
            if not year_data.empty and year >= start_year and year <= end_year:
                # 正常存续期间的数据
                factor_values = year_data.iloc[0, 2:].values  # 排除company_id和year列
                timeline_data[i, :] = factor_values
            elif year < start_year:
                # 企业尚未设立，使用零值或行业平均值
                timeline_data[i, :] = 0.0
            elif year > end_year:
                # 企业已消滅，使用最后一年的数据或零值
                timeline_data[i, :] = timeline_data[i-1, :] if i > 0 else 0.0
        
        # 标准化处理
        timeline_data = scaler.fit_transform(timeline_data)
        
        return torch.tensor(timeline_data, dtype=torch.float32)
    
    def train_survival_model(self, train_loader: torch.utils.data.DataLoader,
                            val_loader: torch.utils.data.DataLoader,
                            epochs: int = 50) -> Dict[str, List[float]]:
        """训练企业存续预测模型"""
        model = self.models['survival_lstm']
        optimizer = self.optimizers['survival_lstm']
        scheduler = self.schedulers['survival_lstm']
        criterion = self.loss_functions['survival']
        
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                survival_target = targets['survival_probability'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(data)
                loss = criterion(outputs['survival_probability'], survival_target)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data = data.to(self.device)
                    survival_target = targets['survival_probability'].to(self.device)
                    
                    outputs = model(data)
                    loss = criterion(outputs['survival_probability'], survival_target)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f'Survival Model Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def train_multitask_model(self, train_loader: torch.utils.data.DataLoader,
                                val_loader: torch.utils.data.DataLoader,
                                epochs: int = 100) -> Dict[str, List[float]]:
        """训练多任务财务指标预测模型"""
        model = self.models['multitask_predictor']
        optimizer = self.optimizers['multitask_predictor']
        scheduler = self.schedulers['multitask_predictor']
        criterion = self.loss_functions['multitask']
        
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data = data.to(self.device)
                    
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step()
            
            if epoch % 20 == 0:
                print(f'Multitask Model Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def train_all_models(self, train_data: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                        val_data: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                        batch_size: int = 32) -> Dict[str, Dict[str, List[float]]]:
        """训练所有深度学习模型"""
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(train_data[0], 
                                                      *[train_data[1][key] for key in train_data[1].keys()])
        val_dataset = torch.utils.data.TensorDataset(val_data[0],
                                                    *[val_data[1][key] for key in val_data[1].keys()])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        results = {}
        
        # 训练企业存续预测模型
        print("Training Survival Prediction Model...")
        results['survival_lstm'] = self.train_survival_model(train_loader, val_loader, epochs=50)
        
        # 训练多任务预测模型
        print("Training Multi-task Financial Predictor...")
        results['multitask_predictor'] = self.train_multitask_model(train_loader, val_loader, epochs=100)
        
        # 训练市场シェア预测模型
        print("Training Market Share Transformer...")
        results['market_transformer'] = self.train_market_transformer(train_loader, val_loader, epochs=75)
        
        # 训练生命周期分类模型
        print("Training Lifecycle Classification CNN...")
        results['lifecycle_cnn'] = self.train_lifecycle_model(train_loader, val_loader, epochs=60)
        
        # 训练要因項目影响分析模型
        print("Training Factor Impact Attention Model...")
        results['factor_attention'] = self.train_factor_impact_model(train_loader, val_loader, epochs=80)
        
        return results
    
    def predict_company_future(self, company_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测企业未来财务状况和生存概率
        Args:
            company_data: (1, sequence_length, input_dim) 单个企业的时序数据
        Returns:
            包含各类预测结果的字典
        """
        predictions = {}
        
        # 设置所有模型为评估模式
        for model in self.models.values():
            model.eval()
        
        with torch.no_grad():
            # 企业存续预测
            survival_pred = self.models['survival_lstm'](company_data)
            predictions['survival'] = survival_pred
            
            # 多任务财务指标预测
            financial_pred = self.models['multitask_predictor'](company_data)
            predictions['financial_metrics'] = financial_pred
            
            # 市场シェア预测
            market_pred = self.models['market_transformer'](company_data)
            predictions['market_share'] = market_pred
            
            # 生命周期分类
            lifecycle_pred = self.models['lifecycle_cnn'](company_data)
            predictions['lifecycle'] = lifecycle_pred
            
            # 要因項目影响分析
            factor_impact_pred = self.models['factor_attention'](company_data)
            predictions['factor_impact'] = factor_impact_pred
        
        return predictions
    
    def analyze_factor_importance(self, companies_data: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        分析23个要因項目的重要度排名
        Args:
            companies_data: (num_companies, sequence_length, input_dim)
        Returns:
            各評価項目的要因項目重要度排名
        """
        self.models['factor_attention'].eval()
        
        with torch.no_grad():
            factor_results = self.models['factor_attention'](companies_data)
        
        importance_rankings = {}
        
        # 分析各評価項目的要因重要度
        for eval_idx in range(9):  # 9个評価項目
            impact_key = f'eval_{eval_idx}_impact'
            if impact_key in factor_results:
                impact_weights = factor_results[impact_key].cpu().numpy()
                
                # 计算平均重要度
                avg_importance = np.mean(impact_weights, axis=0)
                
                # 排序获得重要度排名
                importance_ranking = np.argsort(avg_importance)[::-1]  # 降序排列
                
                importance_rankings[f'evaluation_{eval_idx}'] = importance_ranking
        
        return importance_rankings
    
    def generate_survival_risk_report(self, company_data: torch.Tensor,
                                    company_names: List[str]) -> pd.DataFrame:
        """
        生成企业生存风险报告
        Args:
            company_data: (num_companies, sequence_length, input_dim)
            company_names: 企业名称列表
        Returns:
            包含生存风险评估的DataFrame
        """
        self.models['survival_lstm'].eval()
        
        with torch.no_grad():
            survival_predictions = self.models['survival_lstm'](company_data)
        
        survival_probs = survival_predictions['survival_probability'].cpu().numpy().flatten()
        
        # 计算风险等级
        risk_levels = []
        for prob in survival_probs:
            if prob >= 0.8:
                risk_levels.append('Low Risk')
            elif prob >= 0.6:
                risk_levels.append('Medium Risk')
            elif prob >= 0.4:
                risk_levels.append('High Risk')
            else:
                risk_levels.append('Critical Risk')
        
        report_df = pd.DataFrame({
            'Company': company_names,
            'Survival_Probability': survival_probs,
            'Risk_Level': risk_levels
        })
        
        return report_df.sort_values('Survival_Probability')
    
    def save_models(self, save_dir: str):
        """保存所有训练好的模型"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            torch.save(model.state_dict(), f'{save_dir}/{model_name}.pth')
            
        # 保存配置
        torch.save(self.config, f'{save_dir}/model_config.pth')
        
        print(f"All models saved to {save_dir}")
    
    def load_models(self, load_dir: str):
        """加载预训练的模型"""
        import os
        
        # 加载配置
        config_path = f'{load_dir}/model_config.pth'
        if os.path.exists(config_path):
            self.config = torch.load(config_path)
        
        # 加载模型权重
        for model_name, model in self.models.items():
            model_path = f'{load_dir}/{model_name}.pth'
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded {model_name}")
        
        print(f"All models loaded from {load_dir}")
    
    def train_market_transformer(self, train_loader: torch.utils.data.DataLoader,
                                val_loader: torch.utils.data.DataLoader,
                                epochs: int = 75) -> Dict[str, List[float]]:
        """训练市场シェア预测Transformer模型"""
        model = self.models['market_transformer']
        optimizer = self.optimizers['market_transformer']
        scheduler = self.schedulers['market_transformer']
        criterion = self.loss_functions['classification']
        
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                # 假设targets中包含市场份额类别标签
                market_target = targets.get('market_category', torch.randint(0, 3, (data.size(0),))).to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(data)
                loss = criterion(outputs['market_share_category'], market_target)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            scheduler.step()
            
            if epoch % 15 == 0:
                print(f'Market Transformer Epoch {epoch}: Train Loss={train_loss:.4f}')
        
        return {'train_losses': train_losses, 'val_losses': []}
    
    def train_lifecycle_model(self, train_loader: torch.utils.data.DataLoader,
                                val_loader: torch.utils.data.DataLoader,
                                epochs: int = 60) -> Dict[str, List[float]]:
        """训练生命周期分类CNN模型"""
        model = self.models['lifecycle_cnn']
        optimizer = self.optimizers['lifecycle_cnn']
        scheduler = self.schedulers['lifecycle_cnn']
        criterion = self.loss_functions['classification']
        
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                # 生命周期阶段标签 (0-4: 创业期、成长期、成熟期、衰退期、转型期)
                lifecycle_target = targets.get('lifecycle_stage', torch.randint(0, 5, (data.size(0),))).to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(data)
                loss = criterion(outputs['lifecycle_stage'], lifecycle_target)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            scheduler.step()
            
            if epoch % 12 == 0:
                print(f'Lifecycle CNN Epoch {epoch}: Train Loss={train_loss:.4f}')
        
        return {'train_losses': train_losses, 'val_losses': []}
    
    def train_factor_impact_model(self, train_loader: torch.utils.data.DataLoader,
                                    val_loader: torch.utils.data.DataLoader,
                                    epochs: int = 80) -> Dict[str, List[float]]:
        """训练要因項目影响分析模型"""
        model = self.models['factor_attention']
        optimizer = self.optimizers['factor_attention']
        scheduler = self.schedulers['factor_attention']
        
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(data)
                
                # 使用重构损失训练注意力模型
                attention_weights = outputs['attention_weights']
                reconstruction_loss = F.mse_loss(attention_weights.sum(dim=1), torch.ones(data.size(0), data.size(2)).to(self.device))
                
                # 添加稀疏性正则化，鼓励模型关注重要的要因項目
                sparsity_loss = torch.mean(torch.sum(attention_weights ** 2, dim=-1))
                
                total_loss = reconstruction_loss + 0.01 * sparsity_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            scheduler.step()
            
            if epoch % 16 == 0:
                print(f'Factor Impact Model Epoch {epoch}: Train Loss={train_loss:.4f}')
        
        return {'train_losses': train_losses, 'val_losses': []}


# 使用示例和工厂函数
def create_a2ai_system(config_params: Dict = None) -> A2AIDeepLearningSystem:
    """
    创建A2AI深度学习系统实例
    Args:
        config_params: 配置参数字典
    Returns:
        配置好的A2AI系统实例
    """
    if config_params is None:
        config_params = {}
    
    config = ModelConfig(**config_params)
    system = A2AIDeepLearningSystem(config)
    
    return system


def demo_usage():
    """演示A2AI系统的使用方法"""
    
    # 创建示例数据（150企业 x 40年 x 23要因項目）
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_companies = 150
    sequence_length = 40
    input_dim = 23
    
    # 模拟财务数据
    financial_data = np.random.randn(num_companies, sequence_length, input_dim)
    
    # 模拟目标数据
    target_data = {
        'sales_revenue': np.random.randn(num_companies, 1),
        'sales_growth_rate': np.random.randn(num_companies, 1),
        'operating_margin': np.random.randn(num_companies, 1),
        'net_profit_margin': np.random.randn(num_companies, 1),
        'roe': np.random.randn(num_companies, 1),
        'value_added_ratio': np.random.randn(num_companies, 1),
        'survival_probability': np.random.rand(num_companies, 1),
        'emergence_success_rate': np.random.rand(num_companies, 1),
        'succession_success_degree': np.random.rand(num_companies, 1),
    }
    
    # 创建A2AI系统
    config_params = {
        'input_dim': input_dim,
        'sequence_length': sequence_length,
        'hidden_dims': [128, 256, 512, 256, 128],
        'num_heads': 8,
        'num_layers': 6,
        'learning_rate': 0.001,
        'batch_size': 16,
        'max_epochs': 50
    }
    
    a2ai_system = create_a2ai_system(config_params)
    
    # 转换数据为张量
    input_tensor = torch.tensor(financial_data, dtype=torch.float32)
    target_tensors = {key: torch.tensor(value, dtype=torch.float32) 
                        for key, value in target_data.items()}
    
    # 数据分割
    train_size = int(0.8 * num_companies)
    train_input = input_tensor[:train_size]
    train_targets = {key: value[:train_size] for key, value in target_tensors.items()}
    
    val_input = input_tensor[train_size:]
    val_targets = {key: value[train_size:] for key, value in target_tensors.items()}
    
    # 训练模型
    print("开始训练A2AI深度学习模型...")
    training_results = a2ai_system.train_all_models(
        (train_input, train_targets),
        (val_input, val_targets),
        batch_size=16
    )
    
    # 预测示例
    print("\n进行企业未来预测...")
    sample_company = input_tensor[:1]  # 第一家企业
    predictions = a2ai_system.predict_company_future(sample_company)
    
    print("预测结果:", predictions.keys())
    
    # 要因項目重要度分析
    print("\n分析要因項目重要度...")
    factor_importance = a2ai_system.analyze_factor_importance(input_tensor)
    
    for eval_name, importance_ranking in factor_importance.items():
        print(f"{eval_name} - 最重要的5个要因項目: {importance_ranking[:5]}")
    
    # 生成生存风险报告
    print("\n生成企业生存风险报告...")
    company_names = [f"Company_{i+1}" for i in range(num_companies)]
    risk_report = a2ai_system.generate_survival_risk_report(input_tensor, company_names)
    
    print("风险最高的10家企业:")
    print(risk_report.head(10))
    
    # 保存模型
    a2ai_system.save_models('./a2ai_models')
    print("\nA2AI模型已保存到 './a2ai_models' 目录")


if __name__ == "__main__":
    demo_usage()
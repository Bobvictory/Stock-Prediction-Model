import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from tqdm import tqdm

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据处理类
class StockDataProcessor:
    def __init__(self, data_path, seq_length=32):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.data = self._load_and_process_data(data_path)
        
    def _load_and_process_data(self, path):
        # 读取数据
        data = pd.read_csv(path, header=0, sep=",", encoding="utf-8")
        
        # 对数变换
        cols_to_log = ["Volume", "Turnover"]
        for col in cols_to_log:
            if col in data.columns:
                data[col] = np.log1p(data[col])
        
        # 不需要归一化日期列
        cols_to_normalize = ["Open", "Close", "High", "Low", "Volume", "Turnover", 
                             "Amplitude", "PriceChange", "TurnoverRate"]
        
        # 保存原始数据用于后续分析
        self.raw_data = data.copy()
        
        # 归一化处理
        data[cols_to_normalize] = self.scaler.fit_transform(data[cols_to_normalize])
        
        return data
    
    def get_volatility(self):
        return self.raw_data.groupby('StockCode')['Close'].std().reset_index().rename(columns={'Close': 'Volatility'})

    def prepare_training_data(self, target_column="Close", target_stocks=None):
        stockcodes = self.data["StockCode"].drop_duplicates().tolist()
        if target_stocks is not None:
            stockcodes = [code for code in stockcodes if code in target_stocks]
        all_train_data = []
        for stockcode in stockcodes:
            stock_data = self.data[self.data["StockCode"] == stockcode].drop(columns=["StockCode", "Date"])
            if len(stock_data) < self.seq_length + 1:
                continue
            stock_array = stock_data.values
            sequences = self._create_sequences(stock_array)
            all_train_data.extend(sequences)
        print(f"训练样本数: {len(all_train_data)}")
        return all_train_data
    
    def _create_sequences(self, data):
        """创建时序数据"""
        sequences = []
        target_idx = self.colname2index["Close"] - 2  # 调整索引，因为已经删除了StockCode和Date列
        
        for i in range(len(data) - self.seq_length):
            seq = data[i:i+self.seq_length]
            label = data[i+self.seq_length, target_idx]  # 预测收盘价
            
            # 转换为PyTorch张量
            seq_tensor = torch.FloatTensor(seq)
            label_tensor = torch.FloatTensor([label])
            
            sequences.append((seq_tensor, label_tensor))
        
        return sequences
    
    def save_scaler(self, path):
        """保存归一化器"""
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
    
    @property
    def colname2index(self):
        """特征名到索引的映射"""
        return {x: i for i, x in enumerate(self.data.columns)}

# 改进的模型架构
class EnhancedCNNBiLSTM(nn.Module):
    def __init__(self, input_size, cnn_out_channels=32, cnn_kernel_size=3, 
                 hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(EnhancedCNNBiLSTM, self).__init__()
        
        # CNN层提取特征
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )
        
        # BiLSTM层处理时序关系
        self.bilstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, 1),
            nn.Tanh()
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, feature_dim]
        batch_size, seq_len, feature_dim = x.size()
        
        # CNN处理
        x_cnn = x.permute(0, 2, 1)  # [batch, feature_dim, seq_len]
        x_cnn = self.cnn(x_cnn)     # [batch, cnn_out_channels, seq_len]
        x_cnn = x_cnn.permute(0, 2, 1)  # [batch, seq_len, cnn_out_channels]
        
        # BiLSTM处理
        lstm_out, _ = self.bilstm(x_cnn)  # [batch, seq_len, hidden_size*2]
        
        # 注意力机制
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.permute(0, 2, 1), lstm_out).squeeze(1)  # [batch, hidden_size*2]
        
        # 输出层
        output = self.fc(context)
        return output

# 训练类
class ModelTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.7)
        
    def train(self, train_loader, epochs=50):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # 使用tqdm显示进度条
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, (batch_X, batch_y) in progress_bar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # 更新进度条信息
                progress_bar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(batch_idx+1):.6f}")
            
            # 学习率衰减
            self.scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        return self.model
    
    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not os.path.exists("./model"):
        os.makedirs("./model")
    print("Processing data...")
    data_processor = StockDataProcessor("./temp/feature.csv", seq_length=32)
    data_processor.save_scaler("./model/feature_scaler.bin")
    volatility_df = data_processor.get_volatility()
    sorted_codes = volatility_df.sort_values('Volatility')['StockCode'].tolist()
    n = len(sorted_codes)
    stage1 = sorted_codes[:n//3]
    stage2 = sorted_codes[n//3:2*n//3]
    stage3 = sorted_codes[2*n//3:]
    print("Stage 1: 训练波动小的股票")
    train_data1 = data_processor.prepare_training_data(target_stocks=stage1)
    train_dataset1 = TensorDataset(
        torch.stack([x for x, _ in train_data1]),
        torch.stack([y for _, y in train_data1])
    )
    train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)
    input_size = train_data1[0][0].size(1)
    model = EnhancedCNNBiLSTM(
        input_size=input_size,
        cnn_out_channels=32,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    trainer = ModelTrainer(model, device, learning_rate=0.001)
    trainer.train(train_loader1, epochs=10)
    print("Stage 2: 加入波动中等的股票")
    train_data2 = data_processor.prepare_training_data(target_stocks=stage1+stage2)
    train_dataset2 = TensorDataset(
        torch.stack([x for x, _ in train_data2]),
        torch.stack([y for _, y in train_data2])
    )
    train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True)
    trainer.train(train_loader2, epochs=10)
    print("Stage 3: 全部股票")
    train_data3 = data_processor.prepare_training_data(target_stocks=stage1+stage2+stage3)
    train_dataset3 = TensorDataset(
        torch.stack([x for x, _ in train_data3]),
        torch.stack([y for _, y in train_data3])
    )
    train_loader3 = DataLoader(train_dataset3, batch_size=32, shuffle=True)
    trainer.train(train_loader3, epochs=10)
    model_path = "./model/final_model.pth"
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
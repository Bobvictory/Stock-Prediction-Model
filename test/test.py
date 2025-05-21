import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
from torch.utils.data import TensorDataset, DataLoader

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据处理类
class StockDataProcessor:
    def __init__(self, data_path, seq_length=32, scaler_path="./model/feature_scaler.bin"):
        self.seq_length = seq_length
        self.scaler = self._load_scaler(scaler_path)
        self.data = self._load_and_process_data(data_path)

    def _load_scaler(self, scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return scaler

    def _load_and_process_data(self, path):
        data = pd.read_csv(path, header=0, sep=",", encoding="utf-8")
        # 重命名列名为英文
        data = data.rename(columns={
            "股票代码": "StockCode",
            "日期": "Date",
            "开盘": "Open",
            "收盘": "Close",
            "最高": "High",
            "最低": "Low",
            "成交量": "Volume",
            "成交额": "Turnover",
            "振幅": "Amplitude",
            "涨跌额": "PriceChange",
            "换手率": "TurnoverRate",
            "涨跌幅": "ChangePercent"
        })
        
        # 首先对原始特征进行归一化
        original_cols_to_normalize = ["Open", "Close", "High", "Low", "Volume", "Turnover", 
                                    "Amplitude", "PriceChange", "TurnoverRate"]
        
        # 对Volume和Turnover进行对数转换
        cols_to_log = ["Volume", "Turnover"]
        for col in cols_to_log:
            if col in data.columns:
                data[col] = np.log1p(data[col])
        
        # 保存原始数据
        self.raw_data = data.copy()
        
        # 对原始特征进行归一化
        data[original_cols_to_normalize] = self.scaler.transform(data[original_cols_to_normalize])
        
        # 添加技术指标
        # 使用原始数据计算技术指标
        raw_close = self.raw_data['Close']
        data['MA5'] = data.groupby('StockCode')['Close'].transform(lambda x: x.rolling(window=5).mean())
        data['MA10'] = data.groupby('StockCode')['Close'].transform(lambda x: x.rolling(window=10).mean())
        
        # 处理NaN值
        data = data.bfill().ffill()
        
        return data

    def prepare_test_data(self):
        stockcodes = self.data["StockCode"].drop_duplicates().tolist()
        all_test_data = []
        stockcode_list = []
        for stockcode in stockcodes:
            stock_data = self.data[self.data["StockCode"] == stockcode].drop(columns=["StockCode", "Date"])
            if len(stock_data) < self.seq_length:
                continue
            stock_array = stock_data.values
            seq = stock_array[-self.seq_length:]
            seq_tensor = torch.FloatTensor(seq)
            all_test_data.append(seq_tensor)
            stockcode_list.append(stockcode)
        return all_test_data, stockcode_list

    @property
    def colname2index(self):
        return {x: i for i, x in enumerate(self.data.columns)}

# 改进的模型架构
class EnhancedCNNBiLSTM(nn.Module):
    def __init__(self, input_size, cnn_out_channels=32, cnn_kernel_size=3, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(EnhancedCNNBiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )
        self.bilstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, 1),
            nn.Tanh()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn(x_cnn)
        x_cnn = x_cnn.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x_cnn)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.permute(0, 2, 1), lstm_out).squeeze(1)
        output = self.fc(context)
        return output

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 数据处理
    data_processor = StockDataProcessor("data/test.csv", seq_length=32)
    test_data, stockcode_list = data_processor.prepare_test_data()
    if len(test_data) == 0:
        print("无可用测试数据！")
        return
    # 构建DataLoader
    test_dataset = TensorDataset(torch.stack(test_data))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # 加载模型
    input_size = test_data[0].size(1)
    model = EnhancedCNNBiLSTM(
        input_size=input_size,
        cnn_out_channels=32,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    model_path = "./model/final_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    # 批量预测
    preds = []
    with torch.no_grad():
        for i, (batch_X,) in enumerate(test_loader):
            batch_X = batch_X.to(device)
            output = model(batch_X)
            preds.append(output.item())
    # 还原Close的归一化
    close_index = ["Open", "Close", "High", "Low", "Volume", "Turnover", "Amplitude", "PriceChange", "TurnoverRate"].index("Close")
    close_min = data_processor.scaler.data_min_[close_index]
    close_max = data_processor.scaler.data_max_[close_index]
    preds_original = [p * (close_max - close_min) + close_min for p in preds]
    # 计算涨跌幅
    raw_data = data_processor.raw_data
    result = []
    for stockcode, pred_close in zip(stockcode_list, preds_original):
        # 找到该股票最后一天的真实收盘价
        last_close = raw_data[raw_data["StockCode"] == stockcode]["Close"].values[-1]
        rate = (pred_close - last_close) / last_close * 100
        result.append((stockcode, rate))
    # 排序并输出
    result = sorted(result, key=lambda x: x[1], reverse=True)
    pred_top_10_max_target = [x[0] for x in result[:10]]
    pred_top_10_min_target = [x[0] for x in result[-10:]]
    out_df = pd.DataFrame({
        "涨幅最大股票代码": pred_top_10_max_target,
        "涨幅最小股票代码": pred_top_10_min_target
    })
    out_df.to_csv("./output/result.csv", index=False)
    print("预测结果已保存到 ./output/result.csv")

if __name__ == "__main__":
    main()

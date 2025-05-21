# Stock-Prediction-Model
Stock Prediction Model

# 股票预测模型 README  

## 一、模型概述  
本模型基于深度学习架构 **EnhancedCNNBiLSTM**，结合卷积神经网络（CNN）、双向长短期记忆网络（BiLSTM）和注意力机制（Attention），用于预测股票次日收盘价并筛选涨跌潜力股。模型通过捕捉股票数据的局部模式、长期依赖和关键时间步特征，实现对股票价格的时序预测。


## 二、数据处理流程  
### 1. 输入数据要求  
- **文件格式**：CSV文件，逗号分隔，编码为UTF-8。  
- **文件路径**：训练数据为`./temp/feature.csv`，测试数据为`./temp/feature_test.csv`。  
- **必要列**：  
  | 类别         | 特征名称               | 处理方式                     |  
  |--------------|------------------------|------------------------------|  
  | **基础特征** | Open（开盘价）         | 归一化（MinMaxScaler）       |  
  |              | Close（收盘价）        | 归一化                      |  
  |              | High（最高价）         | 归一化                      |  
  |              | Low（最低价）          | 归一化                      |  
  |              | Volume（成交量）       | 对数转换（log1p）+ 归一化   |  
  |              | Turnover（成交额）     | 对数转换 + 归一化           |  
  |              | Amplitude（振幅）      | 归一化                      |  
  |              | PriceChange（涨跌额）  | 归一化                      |  
  |              | TurnoverRate（换手率） | 归一化                      |  
  | **技术指标** | MA5（5日移动平均线）   | 未归一化                    |  
  |              | MA10（10日移动平均线）  | 未归一化                    |  
  |              | ChangePercent（涨跌幅）| 未归一化                    |  

- **数据格式要求**：  
  - 按时间顺序排列，每只股票至少包含 **32天** 历史数据（对应模型输入窗口长度）。  
  - 示例数据结构：  
    ```csv  
    StockCode,Date,Open,Close,High,Low,Volume,Turnover,Amplitude,PriceChange,TurnoverRate,MA5,MA10,ChangePercent  
    000001,2023-01-01,10.1,10.2,10.3,10.0,1000000,10200000,3.0,0.1,1.0,10.2,10.0,1.0  
    ...  
    ```  


### 2. 数据预处理步骤  
1. **对数转换**：对`Volume`和`Turnover`执行`np.log1p`转换，压缩长尾分布。  
2. **归一化**：使用`MinMaxScaler`对基础特征归一化，保留技术指标原始值。  
3. **序列构建**：按股票分组，取最后32天数据作为输入序列，预测下一日收盘价。  


## 三、模型结构（EnhancedCNNBiLSTM）  
### 1. 网络架构  
```python  
class EnhancedCNNBiLSTM(nn.Module):  
    def __init__(self, input_size, cnn_out_channels=32, cnn_kernel_size=3, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):  
        super().__init__()  
        # 1. CNN层：提取局部特征  
        self.cnn = nn.Sequential(  
            nn.Conv1d(input_size, cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),  
            nn.BatchNorm1d(cnn_out_channels),  
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)  
        )  
        # 2. BiLSTM层：捕捉双向时序依赖  
        self.bilstm = nn.LSTM(  
            input_size=cnn_out_channels,  
            hidden_size=hidden_size,  
            num_layers=num_layers,  
            batch_first=True,  
            dropout=dropout,  
            bidirectional=True  
        )  
        # 3. 注意力机制：加权关键时间步  
        self.attention = nn.Sequential(  
            nn.Linear(hidden_size*2, 1),  
            nn.Tanh()  
        )  
        # 4. 全连接层：输出预测  
        self.fc = nn.Sequential(  
            nn.Linear(hidden_size*2, 64),  
            nn.ReLU(),  
            nn.Dropout(dropout),  
            nn.Linear(64, output_size)  
        )  

    def forward(self, x):  
        # 输入形状：(batch, seq_len, feature_dim)  
        x = x.permute(0, 2, 1)  # 转换为CNN输入格式：(batch, feature_dim, seq_len)  
        x = self.cnn(x)         # CNN特征提取  
        x = x.permute(0, 2, 1)  # 转换为LSTM输入格式：(batch, seq_len, feature_dim)  
        lstm_out, _ = self.bilstm(x)  # BiLSTM输出  
        attn_weights = self.attention(lstm_out)  # 计算注意力权重  
        attn_weights = torch.softmax(attn_weights, dim=1)  
        context = torch.bmm(attn_weights.permute(0, 2, 1), lstm_out).squeeze(1)  # 加权求和  
        output = self.fc(context)  # 最终预测  
        return output  
```  

### 2. 关键组件功能  
| 组件       | 作用                                   |  
|------------|----------------------------------------|  
| **CNN层**  | 提取价格序列的局部模式（如短期波动）   |  
| **BiLSTM层**| 捕捉过去和未来的双向时序依赖关系       |  
| **注意力机制**| 动态加权关键时间步（如重大事件日）     |  
| **全连接层**| 整合特征并输出收盘价预测值             |  


## 四、预测流程  
### 1. 步骤概览  
```mermaid  
graph LR  
A[输入测试数据] --> B[数据预处理]  
B --> C[格式化为32天序列]  
C --> D[模型预测]  
D --> E[反归一化处理]  
E --> F[计算涨跌幅]  
F --> G[排序输出结果]  
```  

### 2. 关键步骤说明  
1. **数据预处理**：  
   - 使用训练阶段保存的`feature_scaler.bin`对测试数据归一化。  
2. **模型预测**：  
   - 输入32天历史数据序列，输出归一化的收盘价预测值。  
3. **结果后处理**：  
   - **反归一化**：将预测值还原为原始尺度：  
     ```python  
     pred_original = pred_normalized * (max - min) + min  
     ```  
   - **计算涨跌幅**：基于预测收盘价与真实收盘价计算：  
     ```python  
     rate = (pred_close - last_true_close) / last_true_close * 100  
     ```  
   - **结果输出**：保存涨幅最大和最小的各10只股票代码至`./output/result.csv`。  


## 五、模型依赖与运行要求  
### 1. 环境依赖  
- **Python包**：  
  ```python  
  pandas>=1.3.3  
  torch>=1.9.0  
  scikit-learn>=1.0.2  
  numpy>=1.21.2  
  ```  
- **硬件**：推荐GPU（如NVIDIA显卡）加速，CPU亦可运行但速度较慢。  

### 2. 文件依赖  
| 文件路径                | 作用                          |  
|-------------------------|-------------------------------|  
| `./model/final_model.pth` | 预训练模型权重文件            |  
| `./model/feature_scaler.bin` | 训练集归一化器参数            |  
| `./temp/feature_test.csv` | 测试数据集                    |  

### 3. 运行命令  
```bash  
python test.py  
```  


## 六、输出结果说明  
- **文件路径**：`./output/result.csv`  
- **内容格式**：  
  | 列名               | 说明                     |  
  |--------------------|--------------------------|  
  | 涨幅最大股票代码   | 预测涨幅前10的股票代码   |  
  | 涨幅最小股票代码   | 预测涨幅后10的股票代码   |  
- **示例输出**：  
  ```csv  
  涨幅最大股票代码,涨幅最小股票代码  
  000005,000008  
  000003,000002  
  ...  
  ```  


## 七、注意事项  
1. **数据完整性**：  
   - 确保测试数据包含所有必要列，且每只股票数据≥32行。  
2. **路径正确性**：  
   - 运行前确认输入文件路径（`./temp/`）和输出目录（`./output/`）存在。  
3. **模型更新**：  
   - 若更换训练数据，需重新训练模型并更新`final_model.pth`和`feature_scaler.bin`。  

 

通过本模型，可基于多维度股票特征和深度学习架构，实现对股票短期走势的量化预测，辅助投资决策。

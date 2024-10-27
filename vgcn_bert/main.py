import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertTokenizer
from torch.optim import AdamW
from sklearn.metrics import classification_report
import pandas as pd
from modeling_vgcn_bert import VGCNBertForSequenceClassification, WordGraphBuilder

# 1. 加载数据
df = pd.read_csv("../processed_spam2.csv")  # 将路径替换为你的文件路径
messages = df['Message'].tolist()
labels = df['Label'].tolist()


# 2. 定义数据集类，进行词嵌入和词图嵌入处理
class MessageDataset(Dataset):
    def __init__(self, messages, labels, tokenizer, graph_builder, max_length=128):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.graph_builder = graph_builder
        self.max_length = max_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = self.messages[idx]
        label = self.labels[idx]

        # 词嵌入
        inputs = self.tokenizer(
            message,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # 构建词图邻接矩阵 (Word Graph) 以及 wgraph_id_to_tokenizer_id_map
        graph_adj, wgraph_id_to_tokenizer_id_map = self.graph_builder([message], tokenizer=self.tokenizer)

        return input_ids, attention_mask, graph_adj, wgraph_id_to_tokenizer_id_map, label


# 3. 初始化 tokenizer 和 word graph 构建器
tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
graph_builder = WordGraphBuilder()

# 创建数据集
dataset = MessageDataset(messages, labels, tokenizer, graph_builder)


# 4. 自定义的 collate_fn，处理稀疏张量的批处理
def custom_collate_fn(batch):
    input_ids, attention_masks, graph_adjs, wgraph_id_to_tokenizer_id_maps, labels = zip(*batch)

    # 将 input_ids 和 attention_mask 堆叠成 dense tensor
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    # 将每个图的稀疏张量放入列表
    graph_adjs = [g.coalesce() for g in graph_adjs]

    # 转换 wgraph_id_to_tokenizer_id_maps 为列表
    wgraph_id_to_tokenizer_id_maps = list(wgraph_id_to_tokenizer_id_maps)

    # 转换 labels 为张量
    labels = torch.tensor(labels)

    return input_ids, attention_masks, graph_adjs, wgraph_id_to_tokenizer_id_maps, labels


# 使用自定义的 collate_fn 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

# 5. 初始化 VGCN-BERT 模型
model = VGCNBertForSequenceClassification.from_pretrained('zhibinlu/vgcn-bert-distilbert-base-uncased', num_labels=2)

# 6. 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# 7. 训练模型
model.train()
for epoch in range(3):  # 假设训练3个epoch
    print(f"Running epoch {epoch + 1}/{3}...")
    for batch in train_loader:
        input_ids, attention_mask, graph_adj, wgraph_id_to_tokenizer_id_maps, labels = batch

        # 将 graph_adj 和 wgraph_id_to_tokenizer_id_maps 作为模型的词图输入
        model.vgcn_bert.embeddings.vgcn.set_wgraphs(graph_adj, wgraph_id_to_tokenizer_id_maps)

        # 模型前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed.")

# 8. 模型评估
model.eval()
predictions, true_labels = [], []
for batch in train_loader:
    input_ids, attention_mask, graph_adj, wgraph_id_to_tokenizer_id_maps, labels = batch

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)

    predictions.extend(preds)
    true_labels.extend(labels)

# 输出分类性能指标
print(classification_report(true_labels, predictions))

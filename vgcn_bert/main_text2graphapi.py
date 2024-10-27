import numpy as np
import pandas as pd
from text2graphapi.src.Cooccurrence import Cooccurrence
from text2graphapi.src.Heterogeneous import Heterogeneous
from text2graphapi.src.IntegratedSyntacticGraph import ISG
import networkx as nx
from transformers import BertTokenizer
import torch
import scipy.sparse as sp
import nltk
from scipy.sparse import block_diag

# 确保下载所需的 nltk 资源
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from vgcn_bert import VGCNBertConfig, VGCNBertForSequenceClassification
from vgcn_bert.modeling_vgcn_bert import _scipy_to_torch, _normalize_adj

# 定义转换图为 SciPy 稀疏矩阵的函数，返回 SciPy 矩阵和映射
def graph_to_scipy_sparse_matrix(graph, tokenizer):
    if isinstance(graph, dict):
        graph = graph.get('graph', graph)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected NetworkX graph, but got {type(graph)}")

    nodes = list(graph.nodes())
    print(f"Number of nodes: {len(nodes)}")

    if len(nodes) == 0:
        print("Empty graph encountered, creating default adjacency and map")
        adj = sp.csr_matrix((1, 1), dtype=np.float32)
        wgraph_id_to_tokenizer_id_map = {0: tokenizer.vocab.get('[UNK]', 0)}
        return adj, wgraph_id_to_tokenizer_id_map

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    row, col, data = [], [], []
    for edge in graph.edges(data=True):
        src = node_to_idx[edge[0]]
        dst = node_to_idx[edge[1]]
        weight = edge[2].get('weight', 1.0)
        row.append(src)
        col.append(dst)
        data.append(weight)

    size = len(nodes)
    adj = sp.coo_matrix((data, (row, col)), shape=(size, size), dtype=np.float32)

    adj = _normalize_adj(adj.tocsr())

    vocab = [idx_to_node[idx] for idx in range(size)]
    wgraph_id_to_tokenizer_id_map = {}
    for graph_id, word in enumerate(vocab):
        wgraph_id_to_tokenizer_id_map[graph_id] = tokenizer.vocab.get(word, tokenizer.vocab.get('[UNK]', 0))

    return adj, dict(sorted(wgraph_id_to_tokenizer_id_map.items()))

# 加载数据和初始化图构建器
data = pd.read_csv('../processed_spam2.csv')
messages = data['Message'].tolist()
labels = data['Label'].tolist()

# 初始化各类图构建器
to_word_coocc_graph = Cooccurrence(
    graph_type='DiGraph',
    language='en',
    apply_prep=True,
    window_size=3,
    output_format='networkx'
)

to_hetero_graph = Heterogeneous(
    graph_type='Graph',
    window_size=20,
    apply_prep=True,
    language='en',
    output_format='networkx'
)

to_isg_graph = ISG(
    graph_type='DiGraph',
    language='en',
    apply_prep=True,
    output_format='networkx'
)

# 准备 Cooccurrence 图的输入数据
corpus_docs = [{'id': idx, 'doc': msg} for idx, msg in enumerate(messages)]

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')

# 初始化 cooccurrence_scipy_adjs 和 cooccurrence_maps
cooccurrence_scipy_adjs = []
cooccurrence_maps = []

# 为每个消息生成 Cooccurrence 图
for doc in corpus_docs:
    graph_output = to_word_coocc_graph.transform([doc], tokenizer)

    if isinstance(graph_output, list):
        cooccurrence_graph = graph_output[0]
    elif isinstance(graph_output, dict):
        cooccurrence_graph = graph_output.get('graph', None)
    else:
        cooccurrence_graph = graph_output  # 直接是图

    # 生成图的 SciPy 稀疏矩阵和映射
    cooccurrence_adj, cooccurrence_map = graph_to_scipy_sparse_matrix(cooccurrence_graph, tokenizer)

    # 将结果添加到列表中
    cooccurrence_scipy_adjs.append(cooccurrence_adj)
    cooccurrence_maps.append(cooccurrence_map)

# 为整个数据集生成 Heterogeneous 图
hetero_graph_output = to_hetero_graph.transform(corpus_docs, tokenizer)
print(f"hetero_graph_output type: {type(hetero_graph_output)}")
if isinstance(hetero_graph_output, list):
    if len(hetero_graph_output) > 0 and isinstance(hetero_graph_output[0], nx.Graph):
        hetero_graph = hetero_graph_output[0]
    elif len(hetero_graph_output) > 0 and isinstance(hetero_graph_output[0], dict):
        hetero_graph = hetero_graph_output[0].get('graph', None)
        if hetero_graph is None:
            raise ValueError("Heterogeneous graph not found in transform output.")
    else:
        raise TypeError("Unknown format for hetero_graph_output.")
elif isinstance(hetero_graph_output, dict):
    hetero_graph = hetero_graph_output.get('graph', None)
    if hetero_graph is None:
        raise ValueError("Heterogeneous graph not found in transform output.")
else:
    hetero_graph = hetero_graph_output  # 直接是图

# 转换 Heterogeneous 图为 SciPy 稀疏矩阵和映射
hetero_adj, hetero_map = graph_to_scipy_sparse_matrix(hetero_graph, tokenizer)
# 归一化并转换为 COO 格式
hetero_adj = _normalize_adj(hetero_adj.tocsr()).tocoo()
# 转换为 PyTorch 稀疏张量
hetero_adj = _scipy_to_torch(hetero_adj)

# 为整个数据集生成 Integrated Syntactic Graph (ISG) 图
isg_graph_output = to_isg_graph.transform(corpus_docs, tokenizer)
print(f"isg_graph_output type: {type(isg_graph_output)}")
if isinstance(isg_graph_output, list):
    if len(isg_graph_output) > 0 and isinstance(isg_graph_output[0], nx.Graph):
        isg_graph = isg_graph_output[0]
    elif len(isg_graph_output) > 0 and isinstance(isg_graph_output[0], dict):
        isg_graph = isg_graph_output[0].get('graph', None)
        if isg_graph is None:
            raise ValueError("ISG graph not found in transform output.")
    else:
        raise TypeError("Unknown format for isg_graph_output.")
elif isinstance(isg_graph_output, dict):
    isg_graph = isg_graph_output.get('graph', None)
    if isg_graph is None:
        raise ValueError("ISG graph not found in transform output.")
else:
    isg_graph = isg_graph_output  # 直接是图

# 转换 ISG 图为 SciPy 稀疏矩阵和映射
isg_adj, isg_map = graph_to_scipy_sparse_matrix(isg_graph, tokenizer)
# 归一化并转换为 COO 格式
isg_adj = _normalize_adj(isg_adj.tocsr()).tocoo()
# 转换为 PyTorch 稀疏张量
isg_adj = _scipy_to_torch(isg_adj)

# 定义重新索引函数
def reindex_mapping(mapping):
    """
    重新索引映射，使键从0开始连续递增。

    参数:
        mapping (dict): 原始的 wgraph_id_to_tokenizer_id_map

    返回:
        dict: 重新索引后的映射
    """
    new_mapping = {new_key: mapping[old_key] for new_key, old_key in enumerate(sorted(mapping.keys()))}
    return new_mapping

# 定义验证函数
def validate_wgraph_id_maps(wgraph_id_to_tokenizer_id_maps):
    """
    验证每个 wgraph_id_to_tokenizer_id_map 是否具有从0到len(map)-1的连续键。

    参数:
        wgraph_id_to_tokenizer_id_maps (list of dict): 映射列表

    抛出:
        ValueError: 如果任何一个映射不满足键连续性要求
    """
    for idx, mapping in enumerate(wgraph_id_to_tokenizer_id_maps):
        if isinstance(mapping, list):
            for sub_mapping in mapping:
                if list(sub_mapping.keys()) != list(range(len(sub_mapping))):
                    raise ValueError(f"Mapping at index {idx} has incorrect keys")
        elif isinstance(mapping, dict):
            if list(mapping.keys()) != list(range(len(mapping))):
                raise ValueError(f"Mapping at index {idx} has incorrect keys")
    print("All maps have been reindexed and validated.")

# 合并 Cooccurrence 图的嵌入
combined_cooccurrence_adj = block_diag(cooccurrence_scipy_adjs).tocoo()

# 归一化邻接矩阵并转换为 COO 格式
combined_cooccurrence_adj = _normalize_adj(combined_cooccurrence_adj.tocsr()).tocoo()

# 转换为 PyTorch 稀疏张量
combined_cooccurrence_adj = _scipy_to_torch(combined_cooccurrence_adj)

# 合并所有 Cooccurrence 映射，并调整 graph_id
combined_cooccurrence_map = {}
current_offset = 0
for sample_map in cooccurrence_maps:
    for graph_id, tok_id in sample_map.items():
        combined_cooccurrence_map[current_offset + graph_id] = tok_id
    current_offset += len(sample_map)

# 重新索引映射
combined_cooccurrence_map = reindex_mapping(combined_cooccurrence_map)
hetero_map = reindex_mapping(hetero_map)
isg_map = reindex_mapping(isg_map)

# 验证映射
wgraph_id_to_tokenizer_id_maps = [combined_cooccurrence_map, hetero_map, isg_map]
validate_wgraph_id_maps(wgraph_id_to_tokenizer_id_maps)

# 组装所有图嵌入
wgraphs = [combined_cooccurrence_adj, hetero_adj, isg_adj]

# 定义自定义数据集类
class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, messages, labels, tokenizer, hetero_adj, hetero_map, isg_adj, isg_map, cooccurrence_adj,
                 cooccurrence_map, max_length=128):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.hetero_adj = hetero_adj
        self.hetero_map = hetero_map
        self.isg_adj = isg_adj
        self.isg_map = isg_map
        self.cooccurrence_adj = cooccurrence_adj
        self.cooccurrence_map = cooccurrence_map
        self.max_length = max_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        text = self.messages[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
        }


# 数据集划分
dataset = SpamDataset(
    messages, labels, tokenizer,
    hetero_adj, hetero_map,
    isg_adj, isg_map,
    combined_cooccurrence_adj, combined_cooccurrence_map
)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 定义自定义的 collate_fn
def custom_collate_fn(batch):
    # 堆叠 input_ids, attention_mask 和 labels
    batch_input_ids = torch.stack([item['input_ids'] for item in batch])
    batch_attention_mask = torch.stack([item['attention_mask'] for item in batch])
    batch_labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask,
        'labels': batch_labels,
    }


# 创建 DataLoader 并使用自定义的 collate_fn
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    collate_fn=custom_collate_fn
)

# 加载 VGCN-BERT 模型并配置图嵌入
config = VGCNBertConfig.from_pretrained('../bert-base-uncased')
model = VGCNBertForSequenceClassification(config, wgraphs, wgraph_id_to_tokenizer_id_maps)

# **不再在训练和评估函数中调用 set_wgraphs**

# 定义训练函数
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    average_loss = total_loss / len(dataloader)
    return average_loss

# 定义评估函数
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = total_correct / total
    return accuracy

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, device)
    val_accuracy = evaluate(model, test_loader, device)
    print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n")

import json
import string
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForTokenClassification, AdamW
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm_


d = {}
d['疾病分类'] = 1
d['等待期'] = 3
d['意外免等待期'] = 5
d['费用报销范围'] = 7
d['公立医院等级'] = 9
d['费用类别'] = 11
d['保障期类型'] = 13
d['最高投保年龄'] = 15
d['最低投保年龄'] = 17
d['事故场景'] = 19
d['豁免原因'] = 21
d['生效类型'] = 23
d['赔付比例'] = 25
d['赔付限额'] = 27
d['赔付次数'] = 29
d['续保免等待期'] = 31
d['津贴计算方式'] = 33
d['单次事故免赔额'] = 35
d['赔付间隔期'] = 37
d['累计天数上限'] = 39
d['医院类型'] = 41
d['延长住院天数'] = 43
d['未成年人赔付规则'] = 45

text = []
entity = []
start = []
end = []
label = []
with open('baoxian/train.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        line = eval(line) # str -> dict
        text.append(line['text']) # text存放了训练集中所有的文本
        entity.append(line['entities'])  # entity存放了训练集中所有的实体


Text = []

for i in range(len(entity)):
    text_ = list(text[i]) # 把text分词
    for j in entity[i]:
        text_[j['start']] = "B-" + j['label']
        for k in range(j['start']+1, j['end']):
            text_[k] = "I-" + j['label']
    Text.append(text_)



for j in range(len(Text)):
   for i in range(len(Text[j])):
        if Text[j][i].startswith('B') and Text[j][i][2::]:
            Text[j][i] = d[Text[j][i][2::]]
        elif Text[j][i].startswith('I') and Text[j][i][2::]:
            Text[j][i] = d[Text[j][i][2::]] + 1
        else:
            Text[j][i] = 0

Text_ = [x for x in Text if len(x) <= 510]
text_ = [x for x in text if len(x) <= 510]

labels = []
for i in range(len(Text_)):
    if len(Text_[i]) <= 512:
        Text_[i] += [-100] * (512-len(Text_[i]))
        labels.append(Text_[i])
labels_tensor = torch.tensor(labels)
# print(labels_tensor.shape) #[494,512]

# L = []
# for i in text:
#     L.append(len(i))
# x = np.arange(1,501)
# plt.plot(x,L)
# plt.title("各样本文本长度")
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_path = "./BERT-CH"
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert = BertModel.from_pretrained(bert_path).to(device)

print("BERT Loaded")
#encoded_input = tokenizer(text_)
# l = []

# for i in encoded_input['input_ids']:
#     l.append(len(i))
#
# for i in range(len(l)):
#     if l[i] > 600:
#         print(i)
# l = np.array(l)
# print(l)
# x = np.arange(1,501)
# plt.bar(x, l)
# plt.show()

encoded_input = tokenizer(text_, return_tensors='pt', padding = 'max_length', truncation = True, max_length = 512)
# print(encoded_input['input_ids'].shape)
# print(encoded_input['attention_mask'].shape)
# print(encoded_input['token_type_ids'].shape)


class PreprocessedNERDataset(Dataset):
    def __init__(self, input, attention_masks, labels):
        self.input = input
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input)


    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }



labels_tensor = torch.tensor(labels)
input_ids = encoded_input['input_ids']
attention_masks = encoded_input['attention_mask']
# print("Inputs size: ", input_ids.shape)
# print("Attention Masks size: ", attention_masks.shape)
# print("Labels size: ", labels_tensor.shape)

dataset = PreprocessedNERDataset(input_ids, attention_masks, labels_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


model_path = './BERT-CH'
model = BertForTokenClassification.from_pretrained(
    model_path,
    num_labels=47  # 根据你的标签数量调整
).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()  # 将模型设置为训练模式
total_loss = 0

start = time.time()
Loss = []
for epoch in range(50):
    total_loss = 0
    for batch in dataloader:
        # 将数据移动到指定的设备上
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 清空之前的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        total_loss += loss.item()
    Loss.append(total_loss)

    print(f'{epoch+1}, {total_loss}')
print(Loss)
end = time.time()
# torch.save(model.state_dict(), 'model_weights_3.pth')
x = np.arange(1,51)

plt.plot(x, Loss)
plt.show()

# print(end-start)

# model = BertForTokenClassification.from_pretrained(
#     model_path,
#     num_labels=47  # 根据你的标签数量调整
# )
# # 用相同的模型结构重新实例化模型
# model.load_state_dict(torch.load('model_weights.pth'))
# model.to(device)  # 不要忘记将模型移到相应的设备
# # 设定模型为评估模式
# model.eval()
#
# # # 初始化计数器
# true_positives = 0
# total_samples = 0
# #
# with torch.no_grad():
#     for batch in dataloader:
#         # 确保数据在正确的设备上
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#
#         # 模型预测
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#
#         # 计算TP
#         for pred, true in zip(predictions, labels):
#             mask = true != -100  # 创建掩码来忽略填充值
#             masked_pred = pred[mask]
#             masked_true = true[mask]
#
#             print("pred:", masked_pred)
#             print("true:", masked_true)
#
#             if torch.equal(masked_pred, masked_true):
#                 true_positives += 1
#             total_samples += 1
#
# print(f"Number of True Positives: {true_positives} out of {total_samples}")





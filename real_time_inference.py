import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import requests
import io

# 你之前定义过的字符集
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
num_classes = len(characters)
num_chars = 5  # 固定长度验证码（你的设置）

# 定义device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 预处理函数：对图像转换为灰度后进行自适应阈值和形态学开运算，
# 最后将结果转换为3通道RGB
def captcha_preprocessing(pil_img):
    np_img = np.array(pil_img)
    # 转为 BGR 格式以供 OpenCV 使用
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    # 转灰度
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    # 自适应阈值（二值化）
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )
    # 形态学操作：开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # 将单通道扩展到3通道
    opened_color = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    processed_pil = Image.fromarray(cv2.cvtColor(opened_color, cv2.COLOR_BGR2RGB))
    return processed_pil

# 假设 AttentionOCR 模型已经定义过
class AttentionDecoder(nn.Module):
    def __init__(self, num_chars, hidden_dim):
        super(AttentionDecoder, self).__init__()
        # learnable queries for each character position (num_chars, hidden_dim)
        self.queries = nn.Parameter(torch.randn(num_chars, hidden_dim))
    
    def forward(self, encoder_outputs):
        # encoder_outputs: [B, T, hidden_dim]
        B, T, D = encoder_outputs.size()
        # expand queries to batch: [B, num_chars, hidden_dim]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_chars, hidden_dim]
        # Compute attention scores: (B, num_chars, T)
        scores = torch.bmm(queries, encoder_outputs.transpose(1, 2))
        attn_weights = torch.softmax(scores, dim=-1)  # [B, num_chars, T]
        # context: weighted sum over time steps
        context = torch.bmm(attn_weights, encoder_outputs)  # [B, num_chars, hidden_dim]
        return context

class AttentionOCR(nn.Module):
    def __init__(self, num_classes, num_chars=5):
        super(AttentionOCR, self).__init__()
        # CNN部分：提取特征，保留时序信息
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出: (B,32,32,100)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出: (B,64,16,50)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,1), (2,1))  # 输出: (B,128,8,50)
        )
        # 将 conv3 输出形状 [B, 128, 8, 50] 变为时序特征
        # 时序长度 T=50, 每个时间步特征维度 = 128*8 = 1024
        self.fc = nn.Linear(128 * 8, 64)  # 降维至64
        # 双向 GRU：输入64，隐藏层128，双向输出维度256
        self.gru = nn.GRU(64, 128, num_layers=1, bidirectional=True, batch_first=True, dropout=0.0)
        # 注意力解码器：生成固定 num_chars 个输出，每个维度为256
        self.attention_decoder = AttentionDecoder(num_chars, hidden_dim=256)
        # 分类器：对每个解码输出映射到字符类别
        self.classifier = nn.Linear(256, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # shape: [B, 128, 8, 50]
        B, C, H, W = x.size()
        # 将特征图转换为时序形式：每个时间步对应宽度方向的一列
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)  # [B, 50, 128*8]
        # 降维到64
        x = self.fc(x)  # [B, 50, 64]
        # GRU编码：输出 [B, 50, 256]
        encoder_outputs, _ = self.gru(x)
        # 注意力解码器：输出 [B, num_chars, 256]
        decoder_outputs = self.attention_decoder(encoder_outputs)
        # 分类器：对每个字符位置输出 logits [B, num_chars, num_classes]
        logits = self.classifier(decoder_outputs)
        logits = self.log_softmax(logits)
        return logits

# 模型实例化
model = AttentionOCR(num_classes, num_chars=num_chars).to(device)
print(model)

# 加载最佳模型参数
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# 图像预处理（必须和训练时完全一致）
img_width, img_height = 200, 64
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

# 在预测时，先调用预处理函数，再应用transform
def preprocess_and_transform(img):
    """
    对输入的 PIL 图像先调用captcha_preprocessing进行预处理，
    再使用transform转换为Tensor
    """
    processed_img = captcha_preprocessing(img)
    return transform(processed_img)

# Beam search decode (你给出的代码)
def beam_search_decode(logits, beam_width=3):
    B, num_chars, num_classes = logits.size()
    all_sequences = []
    for b in range(B):
        sample_logits = logits[b]
        topk_vals, topk_idx = sample_logits.topk(beam_width, dim=1)
        candidates = [[]]
        candidate_scores = [0.0]
        for pos in range(num_chars):
            new_candidates = []
            new_scores = []
            for cand, score in zip(candidates, candidate_scores):
                for i in range(beam_width):
                    new_candidates.append(cand + [topk_idx[pos, i].item()])
                    new_scores.append(score + topk_vals[pos, i].item())
            candidates = new_candidates
            candidate_scores = new_scores
        best_idx = np.argmax(candidate_scores)
        best_sequence = candidates[best_idx]
        pred_text = "".join([characters[i] for i in best_sequence])
        all_sequences.append(pred_text)
    return all_sequences

# 单图像预测函数
def predict_captcha(img_path, model, transform, beam_width=3):
    img = Image.open(img_path).convert('RGB')
    # 调用预处理函数，再转换为Tensor
    img_tensor = preprocess_and_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)  # 输出 [1, num_chars, num_classes]
        preds = beam_search_decode(logits, beam_width=beam_width)
    return preds[0]

# 批量预测函数（更高效）
def batch_predict_captcha(img_paths, model, transform, beam_width=3):
    imgs = []
    for path in img_paths:
        img = Image.open(path).convert('RGB')
        imgs.append(preprocess_and_transform(img))
    imgs_tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        logits = model(imgs_tensor)
        preds = beam_search_decode(logits, beam_width=beam_width)
    return preds

# 使用 requests 库直接调用 captcha 服务器的接口进行实时推理
def captcha_request_inference(captcha_url, beam_width=3):
    """
    通过 HTTP 请求获取 CAPTCHA 图片，并使用模型进行推理。
    
    参数:
      captcha_url: CAPTCHA 图片的 URL
      beam_width: Beam search 的宽度
    返回:
      预测的 CAPTCHA 文本
    """
    response = requests.get(captcha_url)
    if response.status_code == 200:
        captcha_image = Image.open(io.BytesIO(response.content)).convert('RGB')
        img_tensor = preprocess_and_transform(captcha_image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            captcha_text = beam_search_decode(logits, beam_width=beam_width)[0]
        print("Predicted Captcha:", captcha_text)
        return captcha_text
    else:
        print("Failed to retrieve CAPTCHA image. Status code:", response.status_code)
        return None

# 使用示例：
if __name__ == "__main__":
    # 单个图片预测示例
    # example_img = 'example_captcha.png'
    # pred = predict_captcha(example_img, model, transform)
    # print(f"Predicted Captcha for {example_img}: {pred}")

    # # 批量预测示例
    # img_list = ['captcha1.png', 'captcha2.png', 'captcha3.png']
    # preds = batch_predict_captcha(img_list, model, transform)
    # for img_path, captcha_pred in zip(img_list, preds):
    #     print(f"{img_path}: {captcha_pred}")

    # 使用 requests 调用 CAPTCHA 接口示例（请替换为你的 CAPTCHA 服务 URL）
    captcha_server_url = 'http://127.0.0.1:5000/captcha'
    captcha_request_inference(captcha_server_url)

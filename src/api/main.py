from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os
import glob

app = FastAPI(title="News Classification API", description="API phân loại tin tức Tiếng Anh sử dụng DistilBERT")

# --- CẤU HÌNH ---
MODEL_NAME = "distilbert-base-uncased"
# Bản đồ nhãn (Phải khớp với lúc train)
LABEL_MAP = {0: 'World 🌍', 1: 'Sports ⚽', 2: 'Business 💼', 3: 'Sci/Tech 🚀'}


# --- HÀM TỰ DÒ TÌM MODEL (Tránh lỗi đường dẫn) ---
def find_model_path():
    print("🕵️‍♂️ Đang đi tìm thư mục chứa model...")
    # Lấy thư mục gốc dự án
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # Tìm file config.json (dấu hiệu nhận biết model)
    matches = glob.glob(f"{project_root}/**/config.json", recursive=True)

    for match in matches:
        if "distilbert-news-classifier" in match:
            model_dir = os.path.dirname(match)
            print(f"✅ ĐÃ TÌM THẤY MODEL TẠI: {model_dir}")
            return model_dir

    return None


# --- LOAD MODEL (Chạy 1 lần khi khởi động API) ---
model_path = find_model_path()
if not model_path:
    raise RuntimeError("❌ Không tìm thấy model đã train! Bạn đã chạy train.py chưa?")

print("🧠 Đang nạp model vào RAM...")
device = "mps" if torch.backends.mps.is_available() else "cpu"  # Tối ưu cho Mac
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
print("🚀 API ĐÃ SẴN SÀNG PHỤC VỤ!")


# --- ĐỊNH NGHĨA DỮ LIỆU ĐẦU VÀO ---
class NewsRequest(BaseModel):
    text: str


# --- API ENDPOINT ---
@app.post("/predict")
async def predict_news(request: NewsRequest):
    """
    Nhận một đoạn văn bản tiếng Anh -> Trả về chủ đề dự đoán.
    """
    try:
        # 1. Chuẩn bị dữ liệu
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

        # 2. Dự đoán (Không tính đạo hàm để tiết kiệm RAM)
        with torch.no_grad():
            outputs = model(**inputs)

        # 3. Lấy kết quả
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # Tính phần trăm tự tin
        prediction_idx = torch.argmax(probs, dim=-1).item()  # Lấy vị trí có điểm cao nhất
        confidence = probs[0][prediction_idx].item()  # Độ tự tin (0.0 -> 1.0)

        label = LABEL_MAP.get(prediction_idx, "Unknown")

        return {
            "topic": label,
            "confidence": f"{confidence:.2%}",  # Chuyển thành phần trăm (ví dụ 95.5%)
            "raw_text": request.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "Hello! Đây là API phân loại tin tức. Hãy gọi endpoint /predict nhé."}
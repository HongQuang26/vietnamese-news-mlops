import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os
import glob

# --- CẤU HÌNH ---
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4


def find_file(filename, search_path):
    print(f"🕵️‍♂️ Đang đi tìm file '{filename}' trong dự án...")
    matches = glob.glob(f"{search_path}/**/{filename}", recursive=True)
    if matches:
        return matches[0]
    return None


def get_data_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    standard_path = os.path.join(project_root, "data", "processed", "news_clean.csv")

    if os.path.exists(standard_path):
        return standard_path

    print(f"⚠️ Không thấy file ở chỗ chuẩn, đang quét tìm...")
    found_path = find_file("news_clean.csv", project_root)
    if found_path:
        print(f"✅ ĐÃ TÌM THẤY! File đang nằm ở: {found_path}")
        return found_path
    return None


def train():
    print("🔥 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN (FULL POWER)...")

    data_path = get_data_path()
    if not data_path:
        print("❌ LỖI: Không tìm thấy dữ liệu!")
        return

    print(f"📂 Đang đọc dữ liệu từ: {data_path}")
    df = pd.read_csv(data_path)

    # --- CHẾ ĐỘ FULL DATA (Không cắt nhỏ) ---
    print(f"📊 Dữ liệu đầu vào thực tế: {len(df)} dòng (Full Dataset)")

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    print("🔤 Đang tải Tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text_clean"], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    print("⚙️ Đang mã hóa dữ liệu...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    print("🧠 Đang tải Model DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(data_path)))
    if "data" not in project_root:
        project_root = os.getcwd()
    model_dir = os.path.join(project_root, "models", "distilbert-news-classifier")

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=2,
        # Đã xóa tham số use_mps_device gây lỗi
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    print("🚀 Đang huấn luyện... (Sẽ mất nhiều thời gian hơn, hãy kiên nhẫn!)")
    trainer.train()

    print("📝 Đang chấm điểm...")
    results = trainer.evaluate()
    print(f"🏆 Kết quả Loss: {results['eval_loss']}")

    print(f"💾 Đang lưu model vào: {model_dir}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("✅ HOÀN TẤT! Model xịn đã sẵn sàng.")


if __name__ == "__main__":
    train()
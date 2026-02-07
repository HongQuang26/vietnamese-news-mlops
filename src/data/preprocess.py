import pandas as pd
import os
import re

RAW_DATA_PATH = "data/raw/news_train.csv"
PROCESSED_DATA_DIR = "data/processed"


def clean_text_english(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace('\\n', ' ')
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_data():
    print(" Đang làm sạch dữ liệu Tiếng Anh")
    if not os.path.exists(RAW_DATA_PATH):
        print("Chưa có file raw.")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    df['text_clean'] = df['text'].apply(clean_text_english)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DATA_DIR, "news_clean.csv")

    final_df = df[['text_clean', 'label', 'label_name']]
    final_df.to_csv(save_path, index=False, encoding='utf-8')

    print(f"Đã lưu dữ liệu sạch vào: {save_path}")
    print("Kết quả:")
    print(final_df.head(3))


if __name__ == "__main__":
    process_data()
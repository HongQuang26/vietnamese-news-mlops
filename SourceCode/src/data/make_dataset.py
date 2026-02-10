import os
import pandas as pd
from datasets import load_dataset

RAW_DATA_DIR = "data/raw"
DATASET_NAME = "ag_news"


def download_data():
    print(f"Đang tải bộ dữ liệu Tiếng Anh: {DATASET_NAME}...")

    try:
        dataset = load_dataset(DATASET_NAME, split='train')
        df = pd.DataFrame(dataset)
        label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
        df['label_name'] = df['label'].map(label_map)
        print(" Các cột:", df.columns)

        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        save_path = os.path.join(RAW_DATA_DIR, "news_train.csv")
        df.to_csv(save_path, index=False, encoding='utf-8')

        print(f"Đã lưu dữ liệu vào: {save_path}")
        print(f"Tổng số bài báo: {len(df)}")
        print("3 dòng đầu tiên:")
        print(df[['label_name', 'text']].head(3))

    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    download_data()
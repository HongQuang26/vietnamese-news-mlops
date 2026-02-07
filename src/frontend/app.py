import streamlit as st
import requests
import time

# --- CẤU HÌNH ---
API_URL = "http://127.0.0.1:8000/predict"

# --- GIAO DIỆN ---
st.set_page_config(page_title="AI News Classifier", page_icon="📰")

st.title("📰 Phân Loại Tin Tức AI")
st.markdown("Hệ thống sử dụng mô hình **DistilBERT** đã được huấn luyện trên bộ dữ liệu **AG News**.")
st.markdown("---")

# Ô nhập liệu
news_text = st.text_area("Nhập nội dung bản tin tiếng Anh vào đây:", height=150,
                         placeholder="Example: Apple just released a new iPhone with amazing AI features...")

# Nút bấm
if st.button("🔍 Phân tích ngay"):
    if not news_text.strip():
        st.warning("⚠️ Vui lòng nhập nội dung trước!")
    else:
        # Hiệu ứng loading cho chuyên nghiệp
        with st.spinner('🤖 AI đang đọc và suy nghĩ...'):
            try:
                # Gửi yêu cầu sang Backend API
                response = requests.post(API_URL, json={"text": news_text})

                # Giả vờ ngủ 0.5s để người dùng kịp nhìn thấy hiệu ứng loading :))
                time.sleep(0.5)

                if response.status_code == 200:
                    result = response.json()
                    topic = result['topic']
                    confidence = result['confidence']

                    # Hiển thị kết quả đẹp mắt
                    st.success("✅ Đã phân tích xong!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Chủ đề dự đoán", value=topic)
                    with col2:
                        st.metric(label="Độ tin cậy", value=confidence)

                    # Thanh hiển thị mức độ tự tin
                    # Chuyển string "99.5%" -> float 0.995
                    conf_val = float(confidence.strip('%')) / 100
                    st.progress(conf_val)

                else:
                    st.error(f"❌ Lỗi từ API: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Không kết nối được với API Backend! Bạn đã chạy lệnh 'uvicorn' chưa?")
            except Exception as e:
                st.error(f"❌ Có lỗi xảy ra: {e}")

# Footer
st.markdown("---")
st.caption("Dự án MLOps thực tập - Phát triển bởi Nguyễn Hồng Quang")
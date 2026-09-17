# Mô hình Hồi quy tuyến tính bị Overfitting (nhánh main)

Bài tập minh họa hiện tượng **Overfitting** trong Machine Learning, bằng cách cố tình xây dựng một mô hình hồi quy quá phức tạp so với lượng dữ liệu huấn luyện.

## Mục tiêu

Chứng minh: khi mô hình quá "linh hoạt" (nhiều tham số) nhưng dữ liệu huấn luyện quá ít, mô hình sẽ **học thuộc luôn cả nhiễu** trong tập train, dẫn đến dự đoán rất tệ trên dữ liệu mới (tập test).

## Cách gây overfitting

1. **Mở rộng đặc trưng bằng đa thức**: từ 2 biến gốc `Area`, `Bedrooms`, tạo thành 17 biến:
   - `Area^1` đến `Area^8`
   - `Bedrooms^1` đến `Bedrooms^8`
   - `Area × Bedrooms`
2. **Chỉ dùng 25 mẫu để huấn luyện** (trong khi có 17 biến đầu vào) — số biến gần bằng số mẫu khiến mô hình dễ "học vẹt".
3. **Không dùng regularization** — không có gì ngăn các hệ số phát triển tự do theo nhiễu của tập train.

## Dữ liệu

- File: `House_Price_Prediction_Dataset.csv`
- Nguồn: [Kaggle](https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset)
- 2000 mẫu, cột: `Id`, `Area`, `Bedrooms`, `Price`
- Chia: 25 mẫu train / 1975 mẫu test (`random seed = 42`)

## Yêu cầu cài đặt

```bash
pip install pandas numpy
```

## Cách chạy

```bash
python main_mo_hinh_overfitting.py
```

(File CSV phải nằm cùng thư mục với file code.)

## Kết quả

| Chỉ số | Train | Test |
|---|---|---|
| MSE | 25,266,685,324.42 | 181,168,557,417.39 |
| R² | 0.4588 | -1.3613 |

## Nhận xét

- Trên tập **train**, mô hình đạt R² = 0.4588 — khớp khá tốt với dữ liệu đã thấy.
- Trên tập **test**, R² = -1.3613 (âm sâu) — dự đoán còn tệ hơn cả việc lấy giá trung bình để đoán đại.
- **Khoảng cách rất lớn giữa hiệu năng train và test** chính là dấu hiệu điển hình của **overfitting**: mô hình không học được quy luật chung, mà chỉ ghi nhớ các đặc điểm riêng (kể cả nhiễu ngẫu nhiên) của 25 mẫu train.

So sánh với nhánh khắc phục overfitting bằng Ridge Regularization: xem nhánh `chong-overfitting` trong repo này.

## Công nghệ sử dụng

- Python 3
- pandas — đọc và xử lý dữ liệu
- NumPy — tính toán vector hóa cho Gradient Descent

## Tác giả

Hoàng Anh

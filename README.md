# Khắc phục Overfitting bằng Ridge Regularization (nhánh chong-overfitting)

Bài tập minh họa cách dùng **Ridge Regularization (L2)** để giảm hiện tượng overfitting đã xây dựng ở nhánh `main`.

## Mục tiêu

Giữ **nguyên** mô hình quá phức tạp và lượng dữ liệu ít ỏi giống hệt nhánh `main` (để so sánh công bằng), nhưng thêm một số hạng "phạt" vào hàm mất mát nhằm ép các hệ số không được quá lớn — từ đó giảm việc mô hình "học vẹt" theo nhiễu của tập train.

## Ridge Regularization là gì

Hàm mất mát gốc (MSE) được cộng thêm số hạng phạt:

```
Hàm mất mát = MSE + lambda * (tổng bình phương các hệ số w)
```

`lambda` càng lớn, mô hình càng bị ép đơn giản. Trong code, điều này chỉ thêm đúng một số hạng vào công thức đạo hàm khi cập nhật trọng số:

```python
dao_ham_w = (2/m) * (X.T @ sai_so) + (2*lambda/m) * trong_so
```

(Hệ số chặn `b` không bị phạt, chỉ phạt các trọng số `w`.)

## Cấu hình giữ giống nhánh main

- 17 biến đa thức (`Area^1..8`, `Bedrooms^1..8`, `Area × Bedrooms`)
- 25 mẫu train / 1975 mẫu test, cùng `random seed = 42`
- Chỉ khác: thêm `lambda = 50.0`

## Dữ liệu

- File: `House_Price_Prediction_Dataset.csv`
- Nguồn: [Kaggle](https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset)

## Yêu cầu cài đặt

```bash
pip install pandas numpy
```

## Cách chạy

```bash
python chong_overfitting_ridge.py
```

(File CSV phải nằm cùng thư mục với file code.)

## Kết quả

| Chỉ số | Train | Test |
|---|---|---|
| MSE | 40,836,359,571.93 | 81,604,497,364.39 |
| R² | 0.1253 | -0.0636 |

## So sánh với nhánh main (không regularization)

| | Nhánh main | Nhánh chong-overfitting |
|---|---|---|
| R² train | 0.4588 | 0.1253 |
| R² test | -1.3613 | -0.0636 |

## Nhận xét

- R² trên **train giảm** (từ 0.4588 xuống 0.1253) — mô hình khớp với dữ liệu train kém hơn trước, đúng như dự đoán, vì bị "ép" đơn giản hơn.
- R² trên **test tăng đáng kể** (từ -1.3613 lên -0.0636, gần về 0) — mô hình dự đoán trên dữ liệu mới ổn định hơn nhiều, không còn "phá hoại" như trước.
- Đây là đánh đổi (trade-off) đặc trưng của regularization: **hy sinh một phần độ khớp trên tập train, để đổi lấy khả năng tổng quát hóa tốt hơn trên dữ liệu chưa từng thấy** — đúng mục tiêu giảm overfitting.

## Công nghệ sử dụng

- Python 3
- pandas — đọc và xử lý dữ liệu
- NumPy — tính toán vector hóa cho Gradient Descent

## Tác giả

Hoàng Anh

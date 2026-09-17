"""
Nhánh: main
Mục tiêu: XÂY DỰNG MÔ HÌNH BỊ OVERFITTING (cố tình)

Cách gây overfitting:
1. Mở rộng 2 biến gốc (Area, Bedrooms) thành 17 biến đa thức
   (Area^1..8, Bedrooms^1..8, Area*Bedrooms) -> mô hình rất "linh hoạt".
2. Chỉ huấn luyện trên 25 mẫu dữ liệu (rất ít so với 17 biến).
3. KHÔNG dùng regularization.

Hệ quả mong đợi: sai số trên tập TRAIN rất thấp (gần như học thuộc),
nhưng sai số trên tập TEST rất cao -> khoảng cách lớn = dấu hiệu overfitting.
"""

import numpy as np
import pandas as pd

BAC_DA_THUC = 8       # bậc cao nhất của đa thức (Area^8, Bedrooms^8)
SO_MAU_TRAIN = 25      # cố tình để rất ít để dễ overfitting
TOC_DO_HOC = 0.1
SO_VONG_LAP = 60000


# =====================================================
# BƯỚC 1: Đọc dữ liệu
# =====================================================
du_lieu = pd.read_csv("House_Price_Prediction_Dataset.csv")

dien_tich = du_lieu["Area"].values.astype(float)
so_phong = du_lieu["Bedrooms"].values.astype(float)
gia_nha = du_lieu["Price"].values.astype(float)

so_mau_tong = len(gia_nha)


# =====================================================
# BƯỚC 2: Chia dữ liệu — train RẤT ÍT, test là phần còn lại
# =====================================================
np.random.seed(42)
chi_so_xao_tron = np.random.permutation(so_mau_tong)

chi_so_train = chi_so_xao_tron[:SO_MAU_TRAIN]
chi_so_test = chi_so_xao_tron[SO_MAU_TRAIN:]

dien_tich_train, dien_tich_test = dien_tich[chi_so_train], dien_tich[chi_so_test]
so_phong_train, so_phong_test = so_phong[chi_so_train], so_phong[chi_so_test]
gia_nha_train, gia_nha_test = gia_nha[chi_so_train], gia_nha[chi_so_test]


# =====================================================
# BƯỚC 3: Chuẩn hóa dữ liệu (tính trên tập train)
# =====================================================
dien_tich_tb, dien_tich_std = dien_tich_train.mean(), dien_tich_train.std()
so_phong_tb, so_phong_std = so_phong_train.mean(), so_phong_train.std()


def chuan_hoa(dt_input, sp_input):
    dt = (dt_input - dien_tich_tb) / dien_tich_std
    sp = (sp_input - so_phong_tb) / so_phong_std
    return dt, sp


dien_tich_train_ch, so_phong_train_ch = chuan_hoa(dien_tich_train, so_phong_train)
dien_tich_test_ch, so_phong_test_ch = chuan_hoa(dien_tich_test, so_phong_test)


# =====================================================
# BƯỚC 4: Tạo các biến đa thức (feature engineering)
# =====================================================
def tao_dac_trung_da_thuc(dt_chuan_hoa, sp_chuan_hoa, bac=BAC_DA_THUC):
    """Biến 2 cột (Area, Bedrooms) thành nhiều cột đa thức."""
    danh_sach_cot = []
    ten_cot = []

    for bac_hien_tai in range(1, bac + 1):
        danh_sach_cot.append(dt_chuan_hoa ** bac_hien_tai)
        ten_cot.append(f"Area^{bac_hien_tai}")

    for bac_hien_tai in range(1, bac + 1):
        danh_sach_cot.append(sp_chuan_hoa ** bac_hien_tai)
        ten_cot.append(f"Bedrooms^{bac_hien_tai}")

    danh_sach_cot.append(dt_chuan_hoa * sp_chuan_hoa)
    ten_cot.append("Area*Bedrooms")

    return np.column_stack(danh_sach_cot), ten_cot


X_train_tho, ten_dac_trung = tao_dac_trung_da_thuc(dien_tich_train_ch, so_phong_train_ch)
X_test_tho, _ = tao_dac_trung_da_thuc(dien_tich_test_ch, so_phong_test_ch)

# Các cột bậc cao (Area^8, Bedrooms^8...) có giá trị rất lớn/rất nhỏ khác nhau
# => phải chuẩn hóa THÊM một lần nữa theo từng cột, nếu không Gradient Descent
# sẽ bị "nổ số" (tràn số, giá trị NaN) vì learning rate không phù hợp cho mọi cột.
trung_binh_dac_trung = X_train_tho.mean(axis=0)
do_lech_chuan_dac_trung = X_train_tho.std(axis=0)

X_train = (X_train_tho - trung_binh_dac_trung) / do_lech_chuan_dac_trung
X_test = (X_test_tho - trung_binh_dac_trung) / do_lech_chuan_dac_trung

so_dac_trung = X_train.shape[1]
print(f"Số mẫu train: {SO_MAU_TRAIN} | Số biến đầu vào (sau khi mở rộng đa thức): {so_dac_trung}")
print("-> Số biến gần bằng/lớn hơn số mẫu train => môi trường lý tưởng để overfitting.\n")


# =====================================================
# BƯỚC 5: Huấn luyện bằng Gradient Descent — KHÔNG regularization
# =====================================================
trong_so = np.zeros(so_dac_trung)   # vector w cho 17 biến
he_so_chan = 0.0                    # b

for vong_lap in range(SO_VONG_LAP):
    gia_du_doan = X_train @ trong_so + he_so_chan
    sai_so = gia_du_doan - gia_nha_train

    dao_ham_w = (2 / SO_MAU_TRAIN) * (X_train.T @ sai_so)
    dao_ham_b = (2 / SO_MAU_TRAIN) * np.sum(sai_so)

    trong_so -= TOC_DO_HOC * dao_ham_w
    he_so_chan -= TOC_DO_HOC * dao_ham_b

    if vong_lap % 4000 == 0:
        mse_train = np.mean(sai_so ** 2)
        print(f"Vòng lặp {vong_lap}: MSE (train) = {mse_train:.2f}")


# =====================================================
# BƯỚC 6: Đánh giá — so sánh TRAIN vs TEST
# =====================================================
def tinh_r2(y_thuc, y_du_doan):
    tong_binh_phuong_sai_so = np.sum((y_thuc - y_du_doan) ** 2)
    tong_binh_phuong_do_lech = np.sum((y_thuc - y_thuc.mean()) ** 2)
    return 1 - (tong_binh_phuong_sai_so / tong_binh_phuong_do_lech)


du_doan_train = X_train @ trong_so + he_so_chan
du_doan_test = X_test @ trong_so + he_so_chan

mse_train = np.mean((du_doan_train - gia_nha_train) ** 2)
mse_test = np.mean((du_doan_test - gia_nha_test) ** 2)
r2_train = tinh_r2(gia_nha_train, du_doan_train)
r2_test = tinh_r2(gia_nha_test, du_doan_test)

print("\n=== KẾT QUẢ: SO SÁNH TRAIN vs TEST ===")
print(f"{'Chỉ số':<10}{'Train':>18}{'Test':>18}")
print(f"{'MSE':<10}{mse_train:>18,.2f}{mse_test:>18,.2f}")
print(f"{'R2':<10}{r2_train:>18.4f}{r2_test:>18.4f}")

print(
    "\nNhận xét: nếu MSE(train) rất thấp và R2(train) gần 1, trong khi MSE(test) "
    "cao vọt và R2(test) rất thấp (hoặc âm sâu) -> đây chính là dấu hiệu OVERFITTING: "
    "mô hình học thuộc tập train (kể cả nhiễu ngẫu nhiên) nhưng không tổng quát hóa được "
    "cho dữ liệu mới."
)


# =====================================================
# BƯỚC 7: Vẽ biểu đồ trực quan
# =====================================================
import matplotlib.pyplot as plt

hinh, cac_truc = plt.subplots(1, 3, figsize=(16, 5))

# Biểu đồ 1: So sánh MSE giữa Train và Test
cac_truc[0].bar(["Train", "Test"], [mse_train, mse_test], color=["steelblue", "salmon"])
cac_truc[0].set_title("So sánh MSE: Train vs Test")
cac_truc[0].set_ylabel("MSE")

# Biểu đồ 2: So sánh R2 giữa Train và Test
cac_truc[1].bar(["Train", "Test"], [r2_train, r2_test], color=["steelblue", "salmon"])
cac_truc[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
cac_truc[1].set_title("So sánh R2: Train vs Test")
cac_truc[1].set_ylabel("R2")

# Biểu đồ 3: Giá thực tế vs Giá dự đoán (càng gần đường chéo càng chính xác)
cac_truc[2].scatter(gia_nha_train, du_doan_train, alpha=0.7, color="steelblue", label="Train")
cac_truc[2].scatter(gia_nha_test, du_doan_test, alpha=0.4, color="salmon", label="Test")
gia_nho_nhat = min(gia_nha.min(), du_doan_train.min(), du_doan_test.min())
gia_lon_nhat = max(gia_nha.max(), du_doan_train.max(), du_doan_test.max())
cac_truc[2].plot([gia_nho_nhat, gia_lon_nhat], [gia_nho_nhat, gia_lon_nhat], "k--", label="Dự đoán hoàn hảo")
cac_truc[2].set_xlabel("Giá thực tế")
cac_truc[2].set_ylabel("Giá dự đoán")
cac_truc[2].set_title("Thực tế vs Dự đoán")
cac_truc[2].legend()

plt.suptitle("Mô hình OVERFITTING (nhánh main) — không regularization")
plt.tight_layout()
plt.savefig("bieu_do_overfitting.png", dpi=150)
print("\nĐã lưu biểu đồ vào file: bieu_do_overfitting.png")
plt.show()

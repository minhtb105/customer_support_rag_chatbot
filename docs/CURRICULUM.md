# Micro-Curriculum: 30 bài tiểu đường + kịch bản voice 7 ngày (VI)

## Tổng quan
30 bài học nhỏ (micro-lessons), mỗi bài nghe khoảng 1 phút qua giọng
người đồng hành đã chọn. Học 1 bài mỗi ngày trong 7 ngày đầu, sau đó học
tiếp theo gợi ý của app (nhóm yếu nhất).

## Nhóm bài (6 nhóm, 30 bài)
- **Hiểu bệnh** (1.1–1.5): bệnh là gì, chỉ số mục tiêu mg/dL, HbA1c, vì sao đo đều, dùng máy đo.
- **Dinh dưỡng** (2.1–2.10): ăn đúng giờ, chia đĩa cơm, bớt ngọt, tinh bột, rau, đạm, béo, trái cây, nước uống, đi tiệc.
- **Vận động** (3.1–3.5): đi bộ 30 phút, khi nào nghỉ (>250 mg/dL), sau ăn, đều đặn, an toàn.
- **Thuốc và theo dõi** (4.1–4.5): uống đúng đơn, ghi chép, tái khám, bảo quản, quên thuốc.
- **Biến chứng** (5.1–5.3): hạ đường huyết (<70 mg/dL, gọi 115), chăm chân, khi nào đi viện (>300 mg/dL).
- **Tâm lý** (6.1–6.2): tinh thần vui vẻ, cả nhà đồng hành.

## Đơn vị
App dùng **mg/dL** thống nhất. Công thức quy đổi từ tài liệu gốc:
`mg/dL = mmol/L × 18` (ví dụ: 7.0 mmol/L ≈ 126 mg/dL, 13.9 ≈ 250, 16.7 ≈ 300).

## 7 ngày đầu
| Ngày | Bài | Thêm |
|------|-----|------|
| 1 | 1.1 + onboarding (tuổi/thuốc/máy đo, bỏ qua được) | sau `/onboarding/voice` |
| 2–7 | 1.2, 2.1, 2.2, 3.1, 4.1, 5.1 | check-in sáng (tái dụng daily-check) + quiz + khen + hẹn mai |

`GET /v1/lessons/today` trả bài theo ngày VN (server-driven). Nghỉ ngày
không bị nhảy cóc: app đưa bài sớm nhất chưa xong.

## Quiz
Nút A/B/C to (chính) + trả lời bằng voice (chỉ nhận chữ A/B/C,
không đoán số). Nói không rõ → bấm nút.

## Giới hạn y khoa
- Nội dung **giáo dục**, không thay lời bác sĩ. Được kể tên thuốc ở mức
  thông tin, **tuyệt đối không** hướng dẫn tăng/giảm/ngưng liều.
- Bài 5.1 (hạ đường huyết) dẫn về đường dây cấp cứu **115** thật.
- Mọi script mới có `status: draft`, chỉ chuyển `reviewed` sau khi
  reviewer đối chiếu BYT QĐ5481 + ADA 2024.

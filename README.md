# 2025_XAI
-- Metrics: nên sử dụng class kế thừa để dễ quản lý các metrics, có chung cấu trúc từ lớp cha 
Vấn đề	nếu k có 
❌ Không có chuẩn chung	Không đảm bảo class nào cũng có __call__() hoặc name
❌ Khó quản lý hàng loạt	Không thể xử lý qua vòng lặp for metric in metrics: một cách an toàn
❌ Không thể kiểm tra thống nhất	Không thể dùng isinstance(metric, BaseMetric) để kiểm tra loại
❌ Khó mở rộng	Không tận dụng được tính kế thừa, phải sửa tay từng class
❌ Khó bảo trì	Đọc code khó biết class nào là "metric", class nào là thứ khác

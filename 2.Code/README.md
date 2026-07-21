# Quadrotor MPC + MuJoCo (plant thật)

Phiên bản kết hợp: **MPC (CasADi/do-mpc/IPOPT) vẫn là "bộ não"** điều khiển — không
đổi so với bản Python thuần trước đó — nhưng bây giờ được điều khiển trên **MuJoCo**,
một physics engine rigid-body độc lập, thay cho simulator nội bộ của do-mpc (vốn
luôn dùng đúng mô hình mà MPC giả định). Nhờ vậy đây là phép kiểm tra thật về độ
**robust của MPC trước sai lệch mô hình (model mismatch)**, cộng thêm **va chạm thật**
(contact-based, không chỉ dựa vào khoảng cách mềm).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

```bash
streamlit run app_mujoco.py
```

### Bật render đẹp bằng MuJoCo (tuỳ chọn)

Tab "Render MuJoCo" cần một backend OpenGL hoạt động. Trên Linux không có màn hình
(headless server), cần set biến môi trường **trước khi khởi động** app:

```bash
MUJOCO_GL=egl streamlit run app_mujoco.py
# hoặc nếu máy không có EGL:
MUJOCO_GL=osmesa streamlit run app_mujoco.py
```

Trên Windows/Mac hoặc Linux có màn hình, thường không cần set gì — MuJoCo tự chọn
backend phù hợp. Nếu backend không khả dụng, app sẽ **tự động báo và chuyển sang**
tab "Bay 3D (Plotly)" (luôn hoạt động, không cần cấu hình gì) mà không bị lỗi/crash.

## Hiệu năng — đã đo đạc và tối ưu

**Đính chính**: nhận định trước đó ("chậm hơn do chi phí tích phân vật lý độc lập
của MuJoCo") là **sai**. Profiling thực tế cho thấy:

| Thành phần | Thời gian/tick |
|---|---|
| MuJoCo (tích phân vật lý, 25 substep) | **~0.36 ms** |
| do-mpc Simulator (bản không dùng MuJoCo) | ~0.9 ms |
| **Giải MPC (IPOPT)** | **~40–70 ms — đây mới là nút thắt cổ chai thật** |

MuJoCo gần như không tốn gì cả — toàn bộ chi phí nằm ở bộ giải IPOPT, và chi phí đó
tăng thêm khi có ràng buộc né vật cản (~+45%) và trong giai đoạn transient (xa goal).

### Đã tối ưu bằng cách nào

1. **Cache bộ điều khiển MPC đã build** (`build_cached_mpc` + `st.cache_resource`
   trong `app_mujoco.py`): trước đây mỗi lần bấm "Chạy mô phỏng" đều build+setup lại
   toàn bộ NLP từ đầu, kể cả khi chỉ đổi điểm A/B (thứ mà do-mpc vốn đã hỗ trợ đổi
   qua tham số thời biến — time-varying parameter — mà **không cần rebuild**). Giờ
   controller chỉ build lại khi đổi ràng buộc actuator/vật cản/margin.
2. **JIT compilation tuỳ chọn** (checkbox "Bật JIT..."): biên dịch NLP sang C, nhanh
   hơn ~30–40%/tick, đổi lại lần build đầu tiên cho mỗi cấu hình chậm hơn (~10–40s).

### Kết quả đo được (kịch bản có 1 vật cản, sau khi cache)

| | Lần chạy đầu (build) | Lần chạy sau (đổi A/B, tái dùng cache) |
|---|---|---|
| Không JIT | ~6.4s (build ~2-4s + mô phỏng) | ~4.4s |
| Có JIT | ~56s (build+compile ~40-50s + mô phỏng) | **~3.3s** |

→ Nếu bạn thử nhiều điểm A/B khác nhau trong cùng một phiên với cùng ràng buộc
actuator/vật cản, bật JIT (chấp nhận build đầu chậm) cho tốc độ tốt nhất về sau.
Nếu chỉ chạy 1 lần, để JIT tắt (mặc định) là hợp lý hơn.

### Hướng tối ưu thêm (chưa triển khai, gợi ý)

- Dùng linear solver nhanh hơn cho IPOPT (`ma27`/`ma57` của HSL) thay vì `mumps`
  mặc định — cần cài thêm thư viện HSL riêng (có giấy phép học thuật miễn phí).
- Giảm mức độ chặt của ràng buộc né vật cản (chấp nhận `soft_constraint` với phạt
  nhẹ hơn) nếu tốc độ quan trọng hơn độ chính xác né vật cản.
- Chạy nhiều kịch bản song song bằng multiprocessing (mỗi tiến trình build cache
  riêng) nếu cần chạy hàng loạt.

## Cấu trúc file

- `quad_mpc_core.py` — **không đổi** so với bản Python thuần: mô hình động lực học
  phi tuyến (quaternion) mà MPC dùng để dự đoán, cùng bộ điều khiển do-mpc/IPOPT.
- `mujoco_plant.py` — dựng model MJCF khớp khối lượng/quán tính với MPC, 4 actuator
  tổng quát hoá (tổng lực đẩy + 3 mô-men thân, khớp trực tiếp quy ước của MPC), vật
  cản tĩnh (geom cố định) và động (mocap body cập nhật theo quy luật hình sin mỗi
  substep), cùng hàm kiểm tra va chạm tiếp xúc thật (`check_collision`).
- `run_coupled.py` — vòng lặp đóng: mỗi tick, MPC tính `u0` bằng mô hình riêng của
  nó → MuJoCo tích phân vật lý thật với `u0` đó → trạng thái mới đọc từ MuJoCo được
  đưa lại cho MPC ở tick sau.
- `app_mujoco.py` — giao diện Streamlit (giữ nguyên bố cục slider như bản trước).
- `viz.py` — tái sử dụng **y nguyên** module vẽ Plotly của bản Python thuần (định
  dạng dữ liệu đầu ra giống hệt nhau nên không cần sửa gì).

## Những gì đã kiểm chứng thực tế (không chỉ viết code suông)

- **Quy ước trạng thái khớp hoàn toàn, không cần chuyển đổi**: đã kiểm chứng bằng
  thực nghiệm rằng quaternion của MuJoCo (dạng `w,x,y,z`), vận tốc góc (hệ quy chiếu
  thân/body frame) và vận tốc tuyến tính (world frame) trong `qpos`/`qvel` của
  free-joint khớp **chính xác** với quy ước mà MPC dùng — không cần biến đổi cơ sở.
- **Cân bằng hover đúng**: áp `thrust = mg`, mô-men = 0 → drone đứng yên tuyệt đối.
- **Hội tụ chính xác** dù plant là MuJoCo (khác mô hình MPC dùng để dự đoán): sai số
  cuối < 0.005 ở kịch bản mặc định, kể cả với vật cản tĩnh/động.
- **Phát hiện được sai lệch của ràng buộc mềm**: thử nghiệm với margin cực nhỏ
  (0.05 m) cho thấy MuJoCo phát hiện **va chạm tiếp xúc thật** dù chỉ số khoảng cách
  mềm không thể hiện rõ mức độ nguy hiểm — đúng giá trị mà việc thêm MuJoCo mang lại.
- **Trường hợp đối xứng hoàn hảo** (vật cản nằm đúng giữa đường thẳng A→B, không có
  bất kỳ lệch trục nào): ngay cả IPOPT cũng có thể mắc kẹt ở điểm yên ngựa (saddle
  point) toán học — hiếm gặp trong thực tế vì cấu hình thực tế gần như không bao giờ
  đối xứng tuyệt đối, nhưng là giới hạn cần biết.

## Giới hạn hiện tại / hướng mở rộng

- 4 actuator MuJoCo là input **tổng quát hoá** (tổng lực đẩy + 3 mô-men), khớp trực
  tiếp quy ước MPC — chưa mô phỏng ma trận phân bổ 4 động cơ riêng lẻ (control
  allocation matrix). Có thể mở rộng bằng 4 actuator riêng (mỗi rotor một lực đẩy)
  cộng ma trận mixer, để mô phỏng gần hơn nữa với quadrotor thật.

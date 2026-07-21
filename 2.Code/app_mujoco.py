"""
app_mujoco.py — Quadrotor MPC + MuJoCo (true plant) Streamlit app.

Run with:  streamlit run app_mujoco.py

The MPC "brain" (CasADi/do-mpc/IPOPT, quad_mpc_core.py) is UNCHANGED from the
plain Python version - it still predicts using its own simplified quaternion
model. What's new here is that the "true" plant being controlled is MuJoCo's
independently-implemented rigid-body physics engine (mujoco_plant.py) instead of
do-mpc's own Simulator - so this version actually tests whether the MPC is robust
to model mismatch, and adds real contact-based collision detection.

Optional: MuJoCo's own offscreen renderer can produce nicer frames than Plotly,
but needs a working OpenGL backend (EGL/OSMesa on headless Linux, usually set via
`export MUJOCO_GL=egl` before launching this app). If unavailable, the app falls
back to the always-available Plotly 3D animation automatically.
"""
import numpy as np
import streamlit as st

from run_coupled import run_coupled_simulation
from quad_mpc_core import build_cached_mpc
from viz import build_figure, build_timeseries_figure

st.set_page_config(page_title="Quadrotor MPC + MuJoCo", layout="wide", page_icon="🚁")


@st.cache_resource(show_spinner="Đang dựng/biên dịch bộ điều khiển MPC (chỉ 1 lần cho mỗi cấu hình)...")
def get_cached_controller(thrust_max, torque_rp_max, torque_yaw_max, obstacles_key, margin, use_jit):
    """
    Cached across Streamlit re-runs, keyed on (bounds, obstacles, margin, use_jit).
    Changing only start/goal sliders reuses this SAME built (and optionally
    JIT-compiled) controller - profiling showed the MPC solve, not MuJoCo, is
    ~99% of per-tick cost, and rebuilding the whole NLP on every "Run" click (as
    an earlier version of this app did) wastefully re-paid that construction cost
    every single time even though do-mpc already supports changing the goal via a
    time-varying parameter with NO rebuild needed.
    """
    bounds = {'thrust': thrust_max, 'torque_rp': torque_rp_max, 'torque_yaw': torque_yaw_max}
    obstacles = [dict(items) for items in obstacles_key]
    return build_cached_mpc(bounds, obstacles, margin=margin, n_horizon=20, dt=0.05,
                             max_iter=60, use_jit=use_jit)


def obstacles_to_key(obstacles):
    """Converts the obstacles list (of dicts) into a hashable key for st.cache_resource."""
    return tuple(tuple(sorted(o.items())) for o in obstacles)

st.markdown("""
<style>
.stApp { background-color: #0a0d16; }
section[data-testid="stSidebar"] { background-color: #131826; }
h1, h2, h3, p, label, .stMarkdown { color: #e9edf5 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🚁 Quadrotor A → B — MPC (do-mpc/IPOPT) + MuJoCo (plant thật)")
st.caption(
    "MPC vẫn dự đoán bằng mô hình đơn giản hoá của nó (không đổi) — nhưng bây giờ "
    "được điều khiển trên **MuJoCo**, một physics engine rigid-body độc lập, khác "
    "với mô hình nội bộ của MPC. Đây là bài kiểm tra thật về độ robust của MPC "
    "trước sai lệch mô hình (model mismatch), cộng thêm phát hiện va chạm thật "
    "(contact-based) thay vì chỉ dựa vào khoảng cách mềm."
)

with st.sidebar:
    st.header("Trạng thái ban đầu (A)")
    x0 = st.slider("x₀ (m)", -5.0, 5.0, 0.0, 0.1)
    y0 = st.slider("y₀ (m)", -5.0, 5.0, 0.0, 0.1)
    z0 = st.slider("z₀ (m)", 0.2, 5.0, 1.0, 0.1)
    roll0 = st.slider("roll₀ φ (°)", -30, 30, 0, 1)
    pitch0 = st.slider("pitch₀ θ (°)", -30, 30, 0, 1)
    yaw0 = st.slider("yaw₀ ψ (°)", -180, 180, 0, 1)

    st.header("Trạng thái goal (B)")
    xg = st.slider("x_g (m)", -5.0, 5.0, 3.0, 0.1)
    yg = st.slider("y_g (m)", -5.0, 5.0, 2.0, 0.1)
    zg = st.slider("z_g (m)", 0.2, 5.0, 2.5, 0.1)
    rollg = st.slider("roll_g φ (°)", -30, 30, 0, 1)
    pitchg = st.slider("pitch_g θ (°)", -30, 30, 0, 1)
    yawg = st.slider("yaw_g ψ (°)", -180, 180, 45, 1)

    st.header("Ràng buộc actuator")
    constraints_on = st.checkbox("Bật ràng buộc actuator", value=True)
    thrust_max = st.slider("Thrust deviation max (N)", 1.0, 15.0, 6.0, 0.5)
    torque_rp_max = st.slider("Torque roll/pitch max (N·m)", 0.005, 0.100, 0.030, 0.005)
    torque_yaw_max = st.slider("Torque yaw max (N·m)", 0.005, 0.060, 0.020, 0.005)

    st.header("Chướng ngại vật tĩnh")
    obsA_on = st.checkbox("Bật chướng ngại vật tĩnh", value=True)
    obsA_x = st.slider("Vật tĩnh: x (m)", -5.0, 5.0, 1.5, 0.1)
    obsA_y = st.slider("Vật tĩnh: y (m)", -5.0, 5.0, 1.0, 0.1)
    obsA_z = st.slider("Vật tĩnh: z (m)", 0.2, 5.0, 2.0, 0.1)
    obsA_r = st.slider("Vật tĩnh: bán kính (m)", 0.2, 1.5, 0.5, 0.1)

    st.header("Chướng ngại vật động")
    obsB_on = st.checkbox("Bật chướng ngại vật động", value=True)
    obsB_x = st.slider("Vật động: tâm x (m)", -5.0, 5.0, 1.5, 0.1)
    obsB_z = st.slider("Vật động: tâm z (m)", 0.2, 5.0, 1.5, 0.1)
    obsB_amp = st.slider("Vật động: biên độ dao động y (m)", 0.2, 4.0, 2.0, 0.1)
    obsB_period = st.slider("Vật động: chu kỳ (s)", 2.0, 15.0, 6.0, 0.5)
    obsB_r = st.slider("Vật động: bán kính (m)", 0.2, 1.5, 0.4, 0.1)

    st.header("Né vật cản")
    margin = st.slider("Khoảng cách an toàn thêm (m)", 0.1, 1.2, 0.3, 0.05)

    st.header("Mô phỏng")
    sim_seconds = st.slider("Thời lượng mô phỏng (s)", 4.0, 20.0, 10.0, 1.0)
    st.header("Hiệu năng")
    use_jit = st.checkbox(
        "Bật JIT compilation cho MPC (nhanh hơn ~30-40%/tick, nhưng lần build đầu "
        "tiên cho mỗi cấu hình chậm hơn ~10-40s — chỉ đáng dùng nếu bạn sẽ thử "
        "nhiều điểm A/B khác nhau với cùng ràng buộc actuator/vật cản)",
        value=False,
    )
    use_mujoco_render = st.checkbox(
        "Render đẹp bằng MuJoCo (cần backend OpenGL, có thể không khả dụng)",
        value=False,
    )
    run_clicked = st.button("▶ Chạy mô phỏng", type="primary", width="stretch")

    with st.expander("Ghi chú kỹ thuật"):
        st.markdown(
            "- MPC (bộ não) **không đổi**: vẫn CasADi/do-mpc/IPOPT với mô hình "
            "quaternion đơn giản hoá của riêng nó.\n"
            "- Plant thật bây giờ là **MuJoCo** (rigid-body engine độc lập, tích "
            "phân RK4 riêng, có va chạm thật) — không phải simulator nội bộ của "
            "do-mpc dùng đúng mô hình MPC giả định.\n"
            "- **Về hiệu năng**: đo đạc thực tế cho thấy MuJoCo chỉ tốn <1ms/tick — "
            "gần như toàn bộ thời gian mỗi tick là bộ giải IPOPT, không phải "
            "MuJoCo. Vì vậy ứng dụng **cache bộ điều khiển MPC đã build/biên dịch** "
            "(qua `st.cache_resource`) và tái sử dụng khi bạn chỉ đổi điểm A/B — "
            "chỉ build lại khi đổi ràng buộc actuator, vật cản, hoặc bật/tắt JIT.\n"
            "- 4 actuator MuJoCo là input tổng quát hoá (tổng lực đẩy + 3 mô-men "
            "thân), khớp trực tiếp với quy ước của MPC — chưa mô phỏng chi tiết "
            "4 động cơ riêng lẻ (control allocation).\n"
            "- Va chạm hiển thị dưới là **va chạm tiếp xúc thật** (MuJoCo contact), "
            "khác với chỉ số khoảng cách mềm — có thể phát hiện vi phạm mà số liệu "
            "khoảng cách một mình không thấy rõ.\n"
            "- Nếu vật cản nằm đúng trên đường thẳng A→B một cách đối xứng hoàn "
            "toàn, ngay cả IPOPT cũng có thể bị kẹt ở điểm yên ngựa (saddle point) "
            "— trường hợp cực hiếm trong thực tế vì hiếm khi đối xứng tuyệt đối."
        )

if run_clicked:
    obstacles = []
    if obsA_on:
        obstacles.append({'type': 'static', 'x': obsA_x, 'y': obsA_y, 'z': obsA_z, 'radius': obsA_r})
    if obsB_on:
        obstacles.append({'type': 'dynamic', 'x': obsB_x, 'z': obsB_z, 'amp': obsB_amp,
                           'period': obsB_period, 'radius': obsB_r})

    if constraints_on:
        bounds = {'thrust': thrust_max, 'torque_rp': torque_rp_max, 'torque_yaw': torque_yaw_max}
    else:
        bounds = {'thrust': 40.0, 'torque_rp': 2.0, 'torque_yaw': 1.0}

    x0_vals = {'x': x0, 'y': y0, 'z': z0,
               'roll': np.deg2rad(roll0), 'pitch': np.deg2rad(pitch0), 'yaw': np.deg2rad(yaw0)}
    goal_pos = {'x': xg, 'y': yg, 'z': zg}
    goal_euler = {'roll': np.deg2rad(rollg), 'pitch': np.deg2rad(pitchg), 'yaw': np.deg2rad(yawg)}

    cached = get_cached_controller(
        bounds['thrust'], bounds['torque_rp'], bounds['torque_yaw'],
        obstacles_to_key(obstacles), margin, use_jit,
    )

    progress_bar = st.progress(0.0, text="Đang giải MPC + mô phỏng MuJoCo...")

    def progress_cb(frac):
        progress_bar.progress(frac, text=f"Đang giải MPC + mô phỏng MuJoCo... {int(frac*100)}%")

    with st.spinner("Đang chạy IPOPT + MuJoCo..."):
        result = run_coupled_simulation(
            x0_vals=x0_vals, goal_pos=goal_pos, goal_euler=goal_euler,
            bounds=bounds, obstacles=obstacles, margin=margin,
            sim_seconds=sim_seconds, mpc_dt=0.05, n_horizon=20, max_iter=60,
            progress_cb=progress_cb,
            capture_frames=use_mujoco_render, render_every=2,
            cached=cached,
        )
    progress_bar.empty()

    st.session_state['result'] = result
    st.session_state['start'] = {'x': x0, 'y': y0, 'z': z0}
    st.session_state['goal'] = goal_pos
    st.session_state['obstacles'] = obstacles

if 'result' in st.session_state:
    result = st.session_state['result']
    start = st.session_state['start']
    goal = st.session_state['goal']
    obstacles = st.session_state['obstacles']

    final_pos = result['pos'][-1]
    dist = np.linalg.norm(final_pos - np.array([goal['x'], goal['y'], goal['z']]))
    min_clear = result['clearance'].min() if obstacles else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Khoảng cách tới goal (cuối)", f"{dist:.3f} m")
    c2.metric("Khoảng cách vật cản nhỏ nhất (mềm)", f"{min_clear:.2f} m" if min_clear is not None else "không có")
    c3.metric("Va chạm thật (MuJoCo contact)", "🔴 CÓ" if result['collided'] else "🟢 Không")
    c4.metric("Thời gian mô phỏng", f"{result['t'][-1]:.1f} s")

    if result['collided']:
        st.error(
            "MuJoCo phát hiện **va chạm tiếp xúc thật** giữa drone và vật cản trong quá trình bay — "
            "ràng buộc mềm của MPC không đủ để tránh hoàn toàn trong tình huống này. "
            "Thử tăng khoảng cách an toàn hoặc nới ràng buộc actuator."
        )

    tabs = st.tabs(["🎥 Bay 3D (Plotly)", "🖼️ Render MuJoCo", "📈 Đồ thị theo thời gian"])
    with tabs[0]:
        fig = build_figure(result, start, goal, obstacles)
        st.plotly_chart(fig, width='stretch')
        st.caption("Kéo thanh trượt hoặc bấm ▶ Play để xem lại toàn bộ chuyến bay.")
    with tabs[1]:
        if result['frames']:
            idx = st.slider("Khung hình", 0, len(result['frames'])-1, 0, key="mj_frame_slider")
            st.image(result['frames'][idx], width='stretch')
            st.caption(f"Khung {idx+1}/{len(result['frames'])} — render trực tiếp bằng MuJoCo.")
        else:
            st.warning(
                "Không render được bằng MuJoCo trên máy này "
                f"(lý do: `{result['render_error']}`). "
                "Trên Linux không có màn hình, thử chạy với biến môi trường "
                "`MUJOCO_GL=egl` hoặc `MUJOCO_GL=osmesa` trước khi khởi động app. "
                "Tab '🎥 Bay 3D (Plotly)' luôn hoạt động không cần cấu hình gì thêm."
            )
    with tabs[2]:
        fig2 = build_timeseries_figure(result, goal)
        st.plotly_chart(fig2, width='stretch')
else:
    st.info("Chỉnh các thông số ở thanh bên trái rồi bấm **▶ Chạy mô phỏng** để bắt đầu.")

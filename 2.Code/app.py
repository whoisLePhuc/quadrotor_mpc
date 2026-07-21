"""
app.py — Quadrotor MPC flight-simulation Streamlit app.

Run with:  streamlit run app.py

Requires: streamlit, casadi, do-mpc, plotly, numpy  (see requirements.txt)
"""
import numpy as np
import streamlit as st

from quad_mpc_core import run_simulation, build_cached_mpc, M, G
from viz import build_figure, build_timeseries_figure

st.set_page_config(page_title="Quadrotor MPC", layout="wide", page_icon="🚁")


@st.cache_resource(show_spinner="Đang dựng/biên dịch bộ điều khiển MPC (chỉ 1 lần cho mỗi cấu hình)...")
def get_cached_controller(thrust_max, torque_rp_max, torque_yaw_max, obstacles_key, margin, use_jit):
    """Cached across Streamlit re-runs, keyed on (bounds, obstacles, margin, use_jit)
    so changing only start/goal sliders reuses the same built controller instead
    of rebuilding the whole NLP every "Run" click."""
    bounds = {'thrust': thrust_max, 'torque_rp': torque_rp_max, 'torque_yaw': torque_yaw_max}
    obstacles = [dict(items) for items in obstacles_key]
    return build_cached_mpc(bounds, obstacles, margin=margin, n_horizon=20, dt=0.05,
                             max_iter=60, use_jit=use_jit)


def obstacles_to_key(obstacles):
    return tuple(tuple(sorted(o.items())) for o in obstacles)

st.markdown("""
<style>
.stApp { background-color: #0a0d16; }
section[data-testid="stSidebar"] { background-color: #131826; }
h1, h2, h3, p, label, .stMarkdown { color: #e9edf5 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🚁 Quadrotor A → B — Nonlinear MPC (CasADi / do-mpc)")
st.caption(
    "Mô hình động lực học phi tuyến đầy đủ (quaternion) · MPC giải bằng IPOPT thật "
    "(không phải bộ giải tự viết) · né vật cản dạng ràng buộc bất đẳng thức mềm."
)

# ---------------------------------------------------------------------------
# Sidebar: 12 pose sliders + actuator limits + obstacles (mirrors the original
# browser/JS version's slider layout for continuity)
# ---------------------------------------------------------------------------
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
    use_jit = st.checkbox(
        "Bật JIT compilation cho MPC (nhanh hơn ~30-40%/tick, build đầu tiên cho "
        "mỗi cấu hình chậm hơn ~10-40s — đáng dùng nếu thử nhiều điểm A/B khác nhau)",
        value=False,
    )
    run_clicked = st.button("▶ Chạy mô phỏng", type="primary", width="stretch")

    with st.expander("Ghi chú kỹ thuật"):
        st.markdown(
            "- Mô hình 13 trạng thái (vị trí, vận tốc, quaternion, vận tốc góc), "
            "phi tuyến đầy đủ, không giả định góc nhỏ.\n"
            "- MPC giải bằng **IPOPT** (interior-point) qua CasADi/do-mpc — solver "
            "tối ưu hoá đã được kiểm chứng, không phải thuật toán tự viết.\n"
            "- Né vật cản là ràng buộc bất đẳng thức **mềm** (soft constraint, phạt "
            "lớn khi vi phạm) chứ không phải hàng rào thế năng — vật cản động được "
            "dự đoán vị trí tương lai chính xác trong suốt horizon.\n"
            "- Nếu roll_g/pitch_g ≠ 0, drone không thể đứng yên tuyệt đối tại goal "
            "(góc nghiêng luôn tạo gia tốc ngang) — đây là giới hạn vật lý thật."
        )

# ---------------------------------------------------------------------------
# Run simulation on button click
# ---------------------------------------------------------------------------
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
        # generous-but-finite bounds: a real motor never has infinite authority,
        # and the nonlinear plant genuinely cannot tolerate unbounded commands.
        bounds = {'thrust': 40.0, 'torque_rp': 2.0, 'torque_yaw': 1.0}

    x0_vals = {'x': x0, 'y': y0, 'z': z0,
               'roll': np.deg2rad(roll0), 'pitch': np.deg2rad(pitch0), 'yaw': np.deg2rad(yaw0)}
    goal_pos = {'x': xg, 'y': yg, 'z': zg}
    goal_euler = {'roll': np.deg2rad(rollg), 'pitch': np.deg2rad(pitchg), 'yaw': np.deg2rad(yawg)}

    cached = get_cached_controller(
        bounds['thrust'], bounds['torque_rp'], bounds['torque_yaw'],
        obstacles_to_key(obstacles), margin, use_jit,
    )

    progress_bar = st.progress(0.0, text="Đang giải MPC / mô phỏng...")

    def progress_cb(frac):
        progress_bar.progress(frac, text=f"Đang giải MPC / mô phỏng... {int(frac*100)}%")

    with st.spinner("Đang chạy IPOPT..."):
        result = run_simulation(
            x0_vals=x0_vals, goal_pos=goal_pos, goal_euler=goal_euler,
            bounds=bounds, obstacles=obstacles, margin=margin,
            sim_seconds=sim_seconds, dt=0.05, n_horizon=20, max_iter=60,
            progress_cb=progress_cb, cached=cached,
        )
    progress_bar.empty()

    st.session_state['result'] = result
    st.session_state['start'] = {'x': x0, 'y': y0, 'z': z0}
    st.session_state['goal'] = goal_pos
    st.session_state['obstacles'] = obstacles

# ---------------------------------------------------------------------------
# Display results (persist across reruns via session_state)
# ---------------------------------------------------------------------------
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
    c2.metric("Vị trí cuối", f"({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
    c3.metric("Khoảng cách vật cản nhỏ nhất", f"{min_clear:.2f} m" if min_clear is not None else "không có")
    c4.metric("Thời gian mô phỏng", f"{result['t'][-1]:.1f} s")

    tab1, tab2 = st.tabs(["🎥 Bay 3D (animation)", "📈 Đồ thị theo thời gian"])
    with tab1:
        fig = build_figure(result, start, goal, obstacles)
        st.plotly_chart(fig, width='stretch')
        st.caption("Kéo thanh trượt hoặc bấm ▶ Play để xem lại toàn bộ chuyến bay.")
    with tab2:
        fig2 = build_timeseries_figure(result, goal)
        st.plotly_chart(fig2, width='stretch')
else:
    st.info("Chỉnh các thông số ở thanh bên trái rồi bấm **▶ Chạy mô phỏng** để bắt đầu.")

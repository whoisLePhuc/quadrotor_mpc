"""
viz.py
======
Builds the animated Plotly 3D figure (quadrotor + trajectory + obstacles) from a
run_simulation() result dict. Kept separate from app.py so this logic can be unit
tested without needing a running Streamlit server.
"""
import numpy as np
import plotly.graph_objects as go
from quad_mpc_core import quat_from_euler, quat_rotate, obstacle_pos_at

ARM_LEN = 0.5


def _quad_arms(pos, euler):
    """Return the two arm line segments (world coords) for a '+' quadrotor shape."""
    roll, pitch, yaw = euler
    q = quat_from_euler(roll, pitch, yaw)
    p_fb = quat_rotate(q, np.array([ARM_LEN, 0, 0]))
    p_lr = quat_rotate(q, np.array([0, ARM_LEN, 0]))
    front = pos + p_fb
    back = pos - p_fb
    left = pos - p_lr
    right = pos + p_lr
    return front, back, left, right


def build_figure(result, start, goal, obstacles, title="Quadrotor MPC flight"):
    pos = result['pos']
    euler = result['euler']
    t = result['t']
    n = len(t)

    fig = go.Figure()

    # 0: full trajectory line (context, shown from the start)
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='lines', line=dict(color='#39d6c0', width=4), name='Quỹ đạo'
    ))

    # 1,2: quadrotor arms (updated per frame)
    front0, back0, left0, right0 = _quad_arms(pos[0], euler[0])
    fig.add_trace(go.Scatter3d(
        x=[front0[0], back0[0]], y=[front0[1], back0[1]], z=[front0[2], back0[2]],
        mode='lines+markers', line=dict(color='#e9edf5', width=8),
        marker=dict(size=4, color='#39d6c0'), name='Cánh trước-sau'
    ))
    fig.add_trace(go.Scatter3d(
        x=[left0[0], right0[0]], y=[left0[1], right0[1]], z=[left0[2], right0[2]],
        mode='lines+markers', line=dict(color='#e9edf5', width=8),
        marker=dict(size=4, color='#39d6c0'), name='Cánh trái-phải'
    ))

    # 3: start marker
    fig.add_trace(go.Scatter3d(
        x=[start['x']], y=[start['y']], z=[start['z']],
        mode='markers', marker=dict(size=6, color='#39d6c0'), name='Điểm A'
    ))
    # 4: goal marker
    fig.add_trace(go.Scatter3d(
        x=[goal['x']], y=[goal['y']], z=[goal['z']],
        mode='markers', marker=dict(size=6, color='#ffb454'), name='Điểm B (goal)'
    ))

    # 5..: obstacles (marker-approximated spheres; dynamic ones animate per frame)
    obs_trace_start = len(fig.data)
    for i, obs in enumerate(obstacles):
        cx, cy, cz = obstacle_pos_at(obs, 0.0)
        color = '#ff8a3d' if obs['type'] == 'static' else '#ff4dd8'
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz], mode='markers',
            marker=dict(size=max(6, obs['radius']*24), color=color, opacity=0.55),
            name=f"Vật cản {'tĩnh' if obs['type']=='static' else 'động'} {i+1}"
        ))

    # frames: update quad arms (traces 1,2), dynamic obstacles, and a growing trail
    frames = []
    dyn_obs_idx = [i for i, o in enumerate(obstacles) if o['type'] == 'dynamic']
    for k in range(n):
        front, back, left, right = _quad_arms(pos[k], euler[k])
        frame_data = [
            go.Scatter3d(x=[front[0], back[0]], y=[front[1], back[1]], z=[front[2], back[2]]),
            go.Scatter3d(x=[left[0], right[0]], y=[left[1], right[1]], z=[left[2], right[2]]),
        ]
        trace_indices = [1, 2]
        for i in dyn_obs_idx:
            cx, cy, cz = obstacle_pos_at(obstacles[i], t[k])
            frame_data.append(go.Scatter3d(x=[cx], y=[cy], z=[cz]))
            trace_indices.append(obs_trace_start + i)
        frames.append(go.Frame(data=frame_data, traces=trace_indices, name=str(k)))
    fig.frames = frames

    # slider + play/pause controls
    steps = [
        dict(method='animate',
             args=[[str(k)], dict(mode='immediate',
                                   frame=dict(duration=0, redraw=True),
                                   transition=dict(duration=0))],
             label=f"{t[k]:.1f}s")
        for k in range(n)
    ]
    sliders = [dict(steps=steps, x=0, y=0, len=1.0,
                     currentvalue=dict(prefix='t = '))]
    updatemenus = [dict(
        type='buttons', showactive=False, x=0, y=1.08, xanchor='left',
        buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=max(1,int(result['dt']*1000)), redraw=True),
                                   fromcurrent=True, transition=dict(duration=0))]),
            dict(label='⏸ Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
        ]
    )]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x (m)', yaxis_title='y (m)', zaxis_title='z (m, altitude)',
            aspectmode='data',
            bgcolor='#0a0d16',
        ),
        paper_bgcolor='#0a0d16', font=dict(color='#e9edf5'),
        sliders=sliders, updatemenus=updatemenus,
        legend=dict(bgcolor='rgba(18,23,36,0.7)'),
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
    )
    return fig


def build_timeseries_figure(result, goal):
    """2D time-series subplot: position, attitude(deg), control inputs."""
    from plotly.subplots import make_subplots
    t = result['t']
    pos = result['pos']
    euler_deg = np.degrees(result['euler'])
    u = result['u']

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=('Vị trí (m)', 'Góc quay (độ)', 'Tín hiệu điều khiển'))
    fig.add_trace(go.Scatter(x=t, y=pos[:,0], name='x', line=dict(color='#39d6c0')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=pos[:,1], name='y', line=dict(color='#ffb454')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=pos[:,2], name='z', line=dict(color='#ff4dd8')), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=euler_deg[:,0], name='roll', line=dict(color='#39d6c0')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=euler_deg[:,1], name='pitch', line=dict(color='#ffb454')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=euler_deg[:,2], name='yaw', line=dict(color='#ff4dd8')), row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=u[:,0], name='thrust_dev', line=dict(color='#39d6c0')), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=u[:,1], name='taux', line=dict(color='#ffb454')), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=u[:,2], name='tauy', line=dict(color='#ff4dd8')), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=u[:,3], name='tauz', line=dict(color='#a78bfa')), row=3, col=1)

    fig.update_layout(height=650, paper_bgcolor='#0a0d16', plot_bgcolor='rgba(18,23,36,0.5)',
                       font=dict(color='#e9edf5'), margin=dict(l=0, r=0, t=40, b=0),
                       legend=dict(bgcolor='rgba(18,23,36,0.7)'))
    return fig

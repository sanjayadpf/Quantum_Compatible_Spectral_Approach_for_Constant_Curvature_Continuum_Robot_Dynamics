#!/usr/bin/env python3
"""
Classical Chebyshev pseudospectral solver for a 5-link constant-curvature robot
using physical link properties and a constant tip wrench, with NO DAMPING.

- Simulates on [0, T] with T = 10 s
- Chebyshev order n = 30
- Plots:
    * Combined angles for all links
    * Combined angular velocities for all links
    * Per-link plots (angle + angular velocity)
- Animates link motion in (x, y), sampled every 0.1 s

Link properties:
    link length L_link = 0.01 m
    link mass   m      = 0.1 kg
    angular stiffness  k = 1 N·m·rad^-1
    damping            D = 0  (no damping)

Tip wrench:
    Fx = Fy = 1 N
    Mz = 0.2 N·m   <-- updated

State:
    x = [phi_1, phi_dot_1, ..., phi_5, phi_dot_5]^T ∈ R^10
"""

import numpy as np


# ==============================================================
# Chebyshev differentiation matrix
# ==============================================================

def cheb_D(n):
    """Chebyshev–Gauss–Lobatto nodes and differentiation matrix D."""
    if n == 0:
        x = np.array([1.0])
        D = np.array([[0.0]])
        return x, D

    k = np.arange(0, n + 1)
    x = np.cos(np.pi * k / n)

    c = np.ones(n + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** k)

    X = np.tile(x, (n + 1, 1))
    dX = X - X.T

    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D -= np.diag(np.sum(D, axis=1))
    return x, D


# ==============================================================
# Build system from physical properties (NO damping)
# ==============================================================

def build_5link_system_physical_nodamping(n=30, T=10.0):
    """
    Build pseudospectral system L x_vec = b_vec for the 5-link model with:

        M Phi_ddot + K Phi = tau     (no damping)
        x = [Phi; Phi_dot], x' = A_global x + f

    Using:
        L_link = 0.01 m, m = 0.1 kg, k = 1 N·m·rad^-1
        Fx = Fy = 1 N, Mz = 0.2 N·m
    """
    N_links = 5
    d = 2 * N_links

    # Chebyshev nodes in s ∈ [-1,1] then mapped to t ∈ [0,T]
    s_nodes, D_s = cheb_D(n)
    D_t = -(2.0 / T) * D_s
    t_nodes = (T / 2.0) * (1.0 - s_nodes)

    # Physical properties
    L_link = 0.01
    mass = 0.1
    k_ang = 1.0

    # Rotational inertia of slender rod about one end: J = (1/3) m L^2
    J = (1.0 / 3.0) * mass * (L_link ** 2)

    # M, K matrices (diagonal, identical links), D = 0
    M = J * np.eye(N_links)
    K = k_ang * np.eye(N_links)
    M_inv = np.linalg.inv(M)

    # First-order A_global:
    #   x = [Phi; Phi_dot],
    #   A_global = [[0, I],
    #               [-M^{-1}K, 0]]
    A11 = np.zeros((N_links, N_links))
    A12 = np.eye(N_links)
    A21 = - M_inv @ K
    A22 = np.zeros((N_links, N_links))

    A_global = np.block([
        [A11, A12],
        [A21, A22]
    ])  # (10x10)

    # Tip wrench → generalized torque tau_phi
    Fx = 1.0
    Fy = 1.0
    Mz = 0.2   # <-- updated

    L_total = N_links * L_link
    dxdphi = 0.0
    dydphi = L_total / 2.0
    tau_phi = Fx * dxdphi + Fy * dydphi + Mz

    tau_vec = tau_phi * np.ones(N_links)

    # f = [0; M^{-1} tau]
    f = np.zeros(d)
    f[N_links:] = M_inv @ tau_vec

    # Pseudospectral operator:
    # (D_t ⊗ I_d) x_vec = (I_t ⊗ A_global) x_vec + (1 ⊗ f)
    # => [D_t ⊗ I_d - I_t ⊗ A_global] x_vec = (1 ⊗ f)
    I_t = np.eye(n + 1)
    I_d = np.eye(d)

    L = np.kron(D_t, I_d) - np.kron(I_t, A_global)
    ones_t = np.ones(n + 1)
    b_forcing = np.kron(ones_t, f)

    # Initial condition x(0) = 0 (rest)
    x0 = np.zeros(d)
    row0 = slice(0, d)
    L[row0, :] = 0.0
    L[row0, row0] = np.eye(d)
    b_forcing[row0] = x0

    return L, b_forcing, t_nodes, d, A_global, f, tau_phi, J, L_link


# ==============================================================
# Forward kinematics for 5-link planar chain
# ==============================================================

def forward_kinematics_5link(phi_vec, L_link):
    """
    Given phi_vec = [phi_1,...,phi_5] and link length L_link,
    compute joint positions (x,y) from base to tip (6 points).
    """
    N_links = 5
    xs = [0.0]
    ys = [0.0]
    theta = 0.0

    for i in range(N_links):
        theta += phi_vec[i]
        x_next = xs[-1] + L_link * np.cos(theta)
        y_next = ys[-1] + L_link * np.sin(theta)
        xs.append(x_next)
        ys.append(y_next)

    return np.array(xs), np.array(ys)


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    # Chebyshev order and time horizon
    n = 30
    T = 10.0

    L, b, t_nodes, d, A_global, f, tau_phi, J, L_link = \
        build_5link_system_physical_nodamping(n=n, T=T)

    # Solve L x_vec = b_vec
    x_vec = np.linalg.solve(L, b)

    # Reshape into time series: (n+1, d)
    X = x_vec.reshape((n + 1, d))
    Phi_all = X[:, 0::2]  # (n+1, 5)

    print("Rotational inertia per link J =", J)
    print("Generalized tip torque tau_phi =", tau_phi)
    print("\nA_global matrix:")
    print(A_global)

    x_T = X[-1, :]
    print("\nFinal state x(T=10) for all 5 links (phi, phi_dot):")
    for i in range(5):
        phi = x_T[2 * i]
        phi_dot = x_T[2 * i + 1]
        print(f"  Link {i+1}: phi = {phi:.6e}, phi_dot = {phi_dot:.6e}")

    # ==========================================================
    # Plots: combined and per-link
    # ==========================================================
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        # 1) Combined angles phi_i(t) (all links in one figure)
        plt.figure(figsize=(8, 5))
        for i in range(5):
            plt.plot(t_nodes, X[:, 2 * i], label=f"Link {i+1}")
        plt.xlabel("Time t (s)")
        plt.ylabel("Angle phi_i (rad)")
        plt.title("Angular Deflection of All Links (No Damping)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # 2) Combined angular velocities phi_dot_i(t)
        plt.figure(figsize=(8, 5))
        for i in range(5):
            plt.plot(t_nodes, X[:, 2 * i + 1], label=f"Link {i+1}")
        plt.xlabel("Time t (s)")
        plt.ylabel("Angular velocity phi_dot_i (rad/s)")
        plt.title("Angular Velocity of All Links (No Damping)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # 3) Per-link figures: angle + velocity subplots
        for i in range(5):
            fig_i, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            # Angle
            ax[0].plot(t_nodes, X[:, 2 * i], linewidth=2)
            ax[0].set_ylabel(f"phi_{i+1} (rad)")
            ax[0].set_title(f"Link {i+1} Dynamics (No Damping)")
            ax[0].grid(True)
            # Velocity
            ax[1].plot(t_nodes, X[:, 2 * i + 1], linewidth=2)
            ax[1].set_xlabel("Time t (s)")
            ax[1].set_ylabel(f"phi_dot_{i+1} (rad/s)")
            ax[1].grid(True)
            plt.tight_layout()

        # ======================================================
        # Animation: sample every 0.1 s and animate link motion
        # ======================================================
        t_frames = np.arange(0.0, T + 1e-9, 0.1)

        def get_phi_at_time(t_query):
            """Interpolate joint angles phi_i at time t_query."""
            phi_vec = np.zeros(5)
            for j in range(5):
                phi_vec[j] = np.interp(t_query, t_nodes, Phi_all[:, j])
            return phi_vec

        fig, ax = plt.subplots(figsize=(5, 5))

        phi0 = get_phi_at_time(0.0)
        xs0, ys0 = forward_kinematics_5link(phi0, L_link)

        line, = ax.plot(xs0, ys0, '-o')
        title = ax.set_title("t = 0.0 s")

        L_total = 5 * L_link
        pad = L_total * 0.5
        ax.set_xlim(-L_total - pad, L_total + pad)
        ax.set_ylim(-L_total - pad, L_total + pad)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

        def update(frame_idx):
            t_f = t_frames[frame_idx]
            phi_vec = get_phi_at_time(t_f)
            xs, ys = forward_kinematics_5link(phi_vec, L_link)
            line.set_data(xs, ys)
            title.set_text(f"Link motion (no damping), t = {t_f:.1f} s")
            return line, title

        ani = FuncAnimation(
            fig,
            update,
            frames=len(t_frames),
            interval=50,
            blit=True
        )
          # ---------------------------------------
        # Save animation as GIF
        # ---------------------------------------

        from matplotlib.animation import PillowWriter

        gif_writer = PillowWriter(fps=20)
        #ani.save("link_motion_classical_5.gif", writer=gif_writer)

        plt.tight_layout()
        plt.show()

        # To save as MP4 (requires ffmpeg), uncomment:
        # ani.save("link_motion_nodamping_T10_dt0p1_Mz0p2.mp4", writer="ffmpeg", fps=20)

    except ImportError:
        print("\nmatplotlib or matplotlib.animation not installed; skipping plots/animation.")

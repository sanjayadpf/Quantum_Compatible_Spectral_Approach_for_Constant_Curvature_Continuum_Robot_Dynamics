#!/usr/bin/env python3
"""
'Quantum-simulated' solver for the 5-link constant-curvature robot
matching the classical script (no damping), WITHOUT using Qiskit.

Same parameters as classical script:
    n = 30
    T = 10.0 s
    link length L_link = 0.01 m
    link mass   m      = 0.1 kg
    angular stiffness  k = 1 N·m·rad^-1
    damping            D = 0
    Fx = Fy = 1 N
    Mz = 0.2 N·m

Process:
    - Build the same pseudospectral system L x = b
    - Solve classically (numpy.linalg.solve)
    - Amplitude-encode x into a 'quantum state' |X>
      (normalized complex vector of length 2^k)
    - Reconstruct a real x_quantum from |X>
    - Compare classical vs 'quantum-simulated' solutions
      and:
        * plot phi_i, phi_dot_i for all links (combined)
        * plot phi_i, phi_dot_i separately for each link
        * plot error in phi_i, phi_dot_i
        * animate classical vs quantum link motion in (x, y)
"""

import numpy as np
from math import ceil, log2


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

    # Chebyshev nodes in s ∈ [-1,1] then mapped to t ∈ [0, T]
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
    Mz = 0.2

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
# Amplitude encoding (no actual quantum simulator)
# ==============================================================

def amplitude_encode(vec):
    """
    Normalize a classical vector 'vec' and pad to 2^k dimension.

    Returns:
        num_qubits, padded_statevector (complex), norm_of_vec
    """
    vec = np.asarray(vec, dtype=complex)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot encode zero vector.")
    v_norm = vec / norm

    dim = len(v_norm)
    num_qubits = ceil(log2(dim))
    full_dim = 2 ** num_qubits

    state = np.zeros(full_dim, dtype=complex)
    state[:dim] = v_norm
    return num_qubits, state, norm


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
    # Match the classical script
    n = 30
    T = 10.0

    # Build system
    L, b, t_nodes, d, A_global, f, tau_phi, J, L_link = \
        build_5link_system_physical_nodamping(n=n, T=T)

    # Classical solve: L x = b
    x_classical = np.linalg.solve(L, b)

    print("L shape:", L.shape)
    print("Rotational inertia J =", J)
    print("Generalized tip torque tau_phi =", tau_phi)

    # Hermitian embedding tilde_L (optional, for theory)
    zero = np.zeros_like(L)
    tilde_L = np.block([[zero, L], [L.T, zero]])
    print("tilde_L shape:", tilde_L.shape)

    # 'Quantum' amplitude encoding of x_classical
    qX, state_X, norm_x = amplitude_encode(x_classical)
    print(f"Amplitude-encoded |X> uses {qX} qubits (dimension {len(state_X)}).")

    # In an ideal QLSA, the resulting statevector is exactly |X>
    psi_X = state_X.copy()

    # Truncate back to original dimension and reconstruct real quantum solution
    psi_trunc = psi_X[:len(x_classical)]
    x_quantum = np.real(psi_trunc * norm_x)

    # Global relative error (as in Eq. (40))
    rel_err = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical)
    print(f"\nRelative L2 error (classical vs 'quantum-simulated') = {rel_err:.3e}")

    # Time-series reshaping
    X_classical = x_classical.reshape((n + 1, d))
    X_quantum = x_quantum.reshape((n + 1, d))

    print("\nFinal state x(T=10) for all 5 links:")
    for i in range(5):
        phi_c = X_classical[-1, 2 * i]
        phi_q = X_quantum[-1, 2 * i]
        phi_dot_c = X_classical[-1, 2 * i + 1]
        phi_dot_q = X_quantum[-1, 2 * i + 1]
        print(f"  Link {i+1}:")
        print(f"    Classical: phi={phi_c:.6e}, phi_dot={phi_dot_c:.6e}")
        print(f"    Quantum  : phi={phi_q:.6e}, phi_dot={phi_dot_q:.6e}")

    # ----------------------------------------------------------
    # Error over time (per link)
    # ----------------------------------------------------------
    # phi errors and phi_dot errors: quantum - classical
    Phi_classical = X_classical[:, 0::2]   # (n+1, 5)
    Phi_quantum = X_quantum[:, 0::2]
    Phi_err = Phi_quantum - Phi_classical  # (n+1, 5)

    PhiDot_classical = X_classical[:, 1::2]
    PhiDot_quantum = X_quantum[:, 1::2]
    PhiDot_err = PhiDot_quantum - PhiDot_classical

    # ==========================================================
    # Plots & animation
    # ==========================================================
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        # -----------------------------
        # Combined plots: all links
        # -----------------------------
        # 1) Angles phi_i(t)
        plt.figure(figsize=(8, 5))
        for i in range(5):
            plt.plot(t_nodes, Phi_classical[:, i],
                     label=f"Link {i+1} classical", linestyle='-')
            plt.plot(t_nodes, Phi_quantum[:, i],
                     label=f"Link {i+1} quantum", linestyle='--')
        plt.xlabel("Time t (s)")
        plt.ylabel("Angle phi_i (rad)")
        plt.title("Angular Deflection of All Links (Classical vs Quantum, No Damping)")
        plt.grid(True)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        # 2) Angular velocities phi_dot_i(t)
        plt.figure(figsize=(8, 5))
        for i in range(5):
            plt.plot(t_nodes, PhiDot_classical[:, i],
                     label=f"Link {i+1} classical", linestyle='-')
            plt.plot(t_nodes, PhiDot_quantum[:, i],
                     label=f"Link {i+1} quantum", linestyle='--')
        plt.xlabel("Time t (s)")
        plt.ylabel("Angular velocity phi_dot_i (rad/s)")
        plt.title("Angular Velocity of All Links (Classical vs Quantum, No Damping)")
        plt.grid(True)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        # -----------------------------
        # NEW: Error plots
        # -----------------------------
        # |Δphi_i(t)| for all links
        plt.figure(figsize=(8, 5))
        for i in range(5):
            plt.plot(t_nodes, Phi_err[:, i], label=f"Link {i+1}")
        plt.xlabel("Time t (s)")
        plt.ylabel("Angle error Δphi_i (rad)")
        plt.title("Angle Error per Link (Quantum − Classical)")
        plt.grid(True)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        # |Δphi_dot_i(t)| for all links
        plt.figure(figsize=(8, 5))
        for i in range(5):
            plt.plot(t_nodes, PhiDot_err[:, i], label=f"Link {i+1}")
        plt.xlabel("Time t (s)")
        plt.ylabel("Angular velocity error Δphi_dot_i (rad/s)")
        plt.title("Angular Velocity Error per Link (Quantum − Classical)")
        plt.grid(True)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        # -----------------------------
        # Per-link plots (separate)
        # -----------------------------
        for i in range(5):
            fig_i, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            # Angle
            ax[0].plot(t_nodes, Phi_classical[:, i],
                       label="Classical", linestyle='-')
            ax[0].plot(t_nodes, Phi_quantum[:, i],
                       label="Quantum", linestyle='--')
            ax[0].set_ylabel(f"phi_{i+1} (rad)")
            ax[0].set_title(f"Link {i+1} Dynamics (Angle, No Damping)")
            ax[0].grid(True)
            ax[0].legend()

            # Velocity
            ax[1].plot(t_nodes, PhiDot_classical[:, i],
                       label="Classical", linestyle='-')
            ax[1].plot(t_nodes, PhiDot_quantum[:, i],
                       label="Quantum", linestyle='--')
            ax[1].set_xlabel("Time t (s)")
            ax[1].set_ylabel(f"phi_dot_{i+1} (rad/s)")
            ax[1].set_title(f"Link {i+1} Dynamics (Angular Velocity, No Damping)")
            ax[1].grid(True)
            ax[1].legend()

            plt.tight_layout()

        # -----------------------------
        # Animation: classical vs quantum
        # -----------------------------
        # Reuse Phi_classical, Phi_quantum
        # Time frames for animation
        t_frames = np.arange(0.0, T + 1e-9, 0.1)

        def get_phi_at_time(t_query, Phi_all):
            """Interpolate joint angles phi_i at time t_query."""
            phi_vec = np.zeros(5)
            for j in range(5):
                phi_vec[j] = np.interp(t_query, t_nodes, Phi_all[:, j])
            return phi_vec

        fig_anim, ax_anim = plt.subplots(figsize=(5, 5))

        phi0_c = get_phi_at_time(0.0, Phi_classical)
        phi0_q = get_phi_at_time(0.0, Phi_quantum)
        xs0_c, ys0_c = forward_kinematics_5link(phi0_c, L_link)
        xs0_q, ys0_q = forward_kinematics_5link(phi0_q, L_link)

        line_c, = ax_anim.plot(xs0_c, ys0_c, '-o', label="Classical")
        line_q, = ax_anim.plot(xs0_q, ys0_q, '--x', label="Quantum")
        title_anim = ax_anim.set_title("t = 0.0 s")

        L_total = 5 * L_link
        pad = L_total * 0.5
        ax_anim.set_xlim(-L_total - pad, L_total + pad)
        ax_anim.set_ylim(-L_total - pad, L_total + pad)
        ax_anim.set_xlabel("x (m)")
        ax_anim.set_ylabel("y (m)")
        ax_anim.set_aspect('equal', adjustable='box')
        ax_anim.grid(True)
        ax_anim.legend()

        def update(frame_idx):
            t_f = t_frames[frame_idx]
            phi_c = get_phi_at_time(t_f, Phi_classical)
            phi_q = get_phi_at_time(t_f, Phi_quantum)
            xs_c, ys_c = forward_kinematics_5link(phi_c, L_link)
            xs_q, ys_q = forward_kinematics_5link(phi_q, L_link)
            line_c.set_data(xs_c, ys_c)
            line_q.set_data(xs_q, ys_q)
            title_anim.set_text(f"Link motion (no damping), t = {t_f:.1f} s")
            return line_c, line_q, title_anim

        ani = FuncAnimation(
            fig_anim,
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
        ani.save("link_motion_classical_vs_quantum.gif", writer=gif_writer)


        plt.tight_layout()
        plt.show()

        # To save the animation as MP4 (requires ffmpeg), uncomment:
        # ani.save("link_motion_classical_vs_quantum_T10_dt0p1_Mz0p2.mp4",
        #          writer="ffmpeg", fps=20)

    except ImportError:
        print("\nmatplotlib or matplotlib.animation not installed; skipping plots/animation.")

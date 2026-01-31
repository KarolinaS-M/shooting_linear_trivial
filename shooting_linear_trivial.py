import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os

# ======================================================
# Page configuration
# ======================================================

st.set_page_config(
    page_title="Shooting Method: Linear Illustrative Example",
    layout="wide"
)

st.title("Shooting Method: Linear Illustrative Example")

st.markdown(
    "We illustrate the basic idea of the shooting method using the linear boundary "
    "value problem"
)

st.latex(r"""
x'(t)=\lambda x(t), \qquad x(T)=x_T,
""")

st.markdown(
    "for which the analytical solution is available. In this special case, the shooting "
    "condition reduces to a simple linear equation in the unknown initial value "
    r"$\theta=x(0)$."
)

# ======================================================
# Sidebar: user inputs
# ======================================================

with st.sidebar:
    st.header("Model parameters")

    lam = st.number_input("λ (lambda)", value=-1.0, format="%.6f")
    T = st.number_input("Terminal time T", value=5.0, format="%.6f")
    x_T = st.number_input("Terminal value x_T", value=1.0, format="%.6f")

    st.markdown("---")
    st.header("Initial guesses")

    theta0 = st.number_input("Initial guess θ₀", value=0.2, format="%.6f")
    theta1 = st.number_input("Initial guess θ₁", value=2.0, format="%.6f")

# ======================================================
# Analytical structure
# ======================================================

def solution(t, theta):
    return theta * np.exp(lam * t)

def F(theta):
    return theta * np.exp(lam * T) - x_T

# exact theta (known in this linear case)
theta_star = x_T * np.exp(-lam * T)

# ======================================================
# Display shooting information
# ======================================================

st.subheader("Shooting evaluation")

st.latex(r"""
F(\theta)=\theta e^{\lambda T}-x_T
""")

st.markdown(
    "The values of the shooting function for the two initial guesses and for the "
    "exact solution are:"
)

st.write(
    {
        "θ₀": theta0,
        "F(θ₀)": F(theta0),
        "θ₁": theta1,
        "F(θ₁)": F(theta1),
        "θ* (exact)": theta_star,
        "F(θ*)": F(theta_star),
    }
)

# ======================================================
# Plot
# ======================================================

t = np.linspace(0, T, 400)

plt.figure(figsize=(8, 5))

# exact solution
plt.plot(
    t,
    solution(t, theta_star),
    color="black",
    linewidth=2,
    label="Exact solution"
)

# shooting paths
plt.plot(
    t,
    solution(t, theta0),
    linestyle="--",
    label=r"Shot $\theta_0$"
)

plt.plot(
    t,
    solution(t, theta1),
    linestyle="--",
    label=r"Shot $\theta_1$"
)

plt.plot(
    t,
    solution(t, theta_star),
    linestyle=":",
    linewidth=3,
    label=r"Final shot $\theta^\*$"
)

plt.scatter(T, x_T, color="red", zorder=5, label=r"$x(T)=x_T$")

plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)

st.pyplot(plt.gcf())

# ======================================================
# Interpretation
# ======================================================

st.info(
    "In this linear example the shooting condition can be solved in closed form, "
    "so the correct initial value is obtained immediately. As a result, only three "
    "trial paths are shown. In general boundary value problems, the shooting method "
    "requires iterative adjustment of the initial condition."
)
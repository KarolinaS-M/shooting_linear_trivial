import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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
x'(t)=\lambda x(t), \qquad x(T)=x_T
""")

st.markdown(
    "In this special linear case, the shooting condition reduces to a simple linear "
    "equation in the unknown initial value θ = x(0). "
    "The example is therefore intended as an illustration of the shooting idea "
    "rather than as a general numerical algorithm."
)

# ======================================================
# Sidebar: user inputs
# ======================================================

with st.sidebar:
    st.header("Model parameters")

    lam = st.number_input("lambda", value=-1.0, format="%.6f")
    T = st.number_input("Terminal time T", value=5.0, format="%.6f")
    x_T = st.number_input("Terminal value x_T", value=1.0, format="%.6f")

    st.markdown("---")
    st.header("Initial guesses")

    theta0 = st.number_input("Initial guess theta_0", value=0.2, format="%.6f")
    theta1 = st.number_input("Initial guess theta_1", value=2.0, format="%.6f")

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
# Shooting information
# ======================================================

st.subheader("Shooting evaluation")

st.latex(r"""
F(\theta) = \theta e^{\lambda T} - x_T
""")

st.write(
    {
        "theta_0": theta0,
        "F(theta_0)": F(theta0),
        "theta_1": theta1,
        "F(theta_1)": F(theta1),
        "theta_star (exact)": theta_star,
        "F(theta_star)": F(theta_star),
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
    label="Shot theta_0"
)

plt.plot(
    t,
    solution(t, theta1),
    linestyle="--",
    label="Shot theta_1"
)

plt.plot(
    t,
    solution(t, theta_star),
    linestyle=":",
    linewidth=3,
    label="Final shot theta_star"
)

plt.scatter(T, x_T, color="red", zorder=5, label="Terminal condition")

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
    "so the correct initial value is obtained immediately. Only three trial paths "
    "are therefore shown. In general boundary value problems, the shooting method "
    "requires iterative adjustment of the initial condition."
)
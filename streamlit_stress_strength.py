# app.py
import numpy as np
import streamlit as st
from scipy import stats
from scipy.integrate import quad
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stress–Strength Interference", layout="centered")
st.title("Stress–Strength Interference — Reliability Calculator")
st.caption("Compute R = P(Strength > Stress) for a choice of distributions.")

# -------- Helpers --------
DIST_NAMES = ["Normal", "Weibull", "Lognormal", "Exponential"]

def dist_params_ui(side_label: str):
    st.subheader(f"{side_label} distribution")
    name = st.selectbox(
        f"Choose {side_label.lower()} distribution",
        DIST_NAMES,
        key=f"{side_label}_dist",
    )

    if name == "Normal":
        # Set different defaults for Stress vs Strength
        default_mu = 10.0 if side_label == "Stress" else 15.0
        default_sigma = 2.0

        mu = st.number_input(
            f"{side_label} Normal mean μ",
            value=default_mu,
            step=0.5,
            key=f"{side_label}_n_mu"
        )
        sigma = st.number_input(
            f"{side_label} Normal std σ (>0)",
            min_value=1e-9,
            value=default_sigma,
            step=0.1,
            key=f"{side_label}_n_sigma"
        )
        return name, {"mu": mu, "sigma": sigma}

    if name == "Weibull":
        k = st.number_input(f"{side_label} Weibull shape k (>0)", min_value=1e-6, value=2.0, step=0.1,
                            key=f"{side_label}_wb_k")
        scale = st.number_input(f"{side_label} Weibull scale λ (>0)", min_value=1e-9, value=10.0, step=0.5,
                                key=f"{side_label}_wb_scale")
        return name, {"k": k, "scale": scale}

    if name == "Lognormal":
        mu = st.number_input(f"{side_label} Lognormal log-mean μ (of ln X)", value=2.0, step=0.1,
                             key=f"{side_label}_ln_mu")
        sigma = st.number_input(f"{side_label} Lognormal log-std σ (>0)", min_value=1e-9, value=0.5, step=0.05,
                                key=f"{side_label}_ln_sigma")
        return name, {"mu": mu, "sigma": sigma}

    if name == "Exponential":
        mean = st.number_input(f"{side_label} Exponential mean (scale) (>0)", min_value=1e-12, value=10.0, step=0.5,
                               key=f"{side_label}_exp_scale")
        return name, {"scale": mean}

    # fallback
    return name, {}


def make_frozen(name: str, params: dict):
    if name == "Weibull":
        # SciPy: weibull_min(c=k, scale=λ)
        return stats.weibull_min(c=params["k"], scale=params["scale"])
    if name == "Normal":
        return stats.norm(loc=params["mu"], scale=params["sigma"])
    if name == "Lognormal":
        # SciPy: lognorm(s=sigma, scale=exp(mu))
        return stats.lognorm(s=params["sigma"], scale=np.exp(params["mu"]))
    if name == "Exponential":
        # SciPy: expon(scale=mean)
        return stats.expon(scale=params["scale"])
    raise ValueError("Unsupported distribution")


def reliability_integral(stress_rv, strength_rv):
    # R = ∫ f_X(x) * (1 - F_Y(x)) dx over (-inf, +inf)
    # SciPy handles the tails with infinite bounds
    integrand = lambda x: stress_rv.pdf(x) * (1.0 - strength_rv.cdf(x))
    val, _ = quad(integrand, -np.inf, np.inf, limit=300, epsabs=1e-9, epsrel=1e-8)
    # numerical guard
    return float(np.clip(val, 0.0, 1.0))


def support_bounds(rvs, q_lo=1e-4, q_hi=1 - 1e-4):
    # Combined plotting window across both distributions
    qs = []
    for rv in rvs:
        try:
            qs.append(rv.ppf([q_lo, q_hi]))
        except Exception:
            qs.append(np.array([rv.mean() - 4 * rv.std(), rv.mean() + 4 * rv.std()]))
    arr = np.vstack(qs)
    lo = float(np.nanmin(arr[:, 0]))
    hi = float(np.nanmax(arr[:, 1]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -10.0, 10.0
    # widen a bit
    span = hi - lo
    return lo - 0.05 * span, hi + 0.05 * span


# -------- UI --------
left, right = st.columns(2)
with left:
    stress_name, stress_params = dist_params_ui("Stress")
with right:
    strength_name, strength_params = dist_params_ui("Strength")

st.divider()

# Optional Monte Carlo
with st.expander("Monte Carlo (optional)", expanded=False):
    n_mc = st.number_input("Samples (0 to skip MC)", min_value=0, value=0, step=10000)
    rng_seed = st.number_input("Random seed", value=42, step=1)

# Custom CSS for yellow button
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #FFD700; /* yellow (gold) */
        color: black; /* text color */
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #FFC300; /* darker yellow on hover */
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Compute button
if st.button("Compute reliability"):
    try:
        stress_rv = make_frozen(stress_name, stress_params)
        strength_rv = make_frozen(strength_name, strength_params)

        R = reliability_integral(stress_rv, strength_rv)

        st.success(f"Reliability R = P(Strength > Stress) = **{R:.6f}**  ({R * 100:.4f}%)")

        # Monte Carlo check
        if n_mc and n_mc > 0:
            rng = np.random.default_rng(int(rng_seed))
            x = stress_rv.rvs(size=int(n_mc), random_state=rng)
            y = strength_rv.rvs(size=int(n_mc), random_state=rng)
            r_mc = np.mean(y > x)
            se = np.sqrt(max(r_mc * (1 - r_mc), 1e-16) / n_mc)
            st.info(f"Monte Carlo estimate with n={int(n_mc):,}: **{r_mc:.6f}** (± {1.96 * se:.6f} at 95%)")

        # Plot PDFs
        lo, hi = support_bounds([stress_rv, strength_rv])
        xs = np.linspace(lo, hi, 800)
        stress_pdf = stress_rv.pdf(xs)
        strength_pdf = strength_rv.pdf(xs)

        fig, ax = plt.subplots()
        ax.plot(xs, stress_pdf, label=f"Stress PDF ({stress_name})")
        ax.plot(xs, strength_pdf, label=f"Strength PDF ({strength_name})")
        ax.set_xlabel("Value")
        ax.set_ylabel("PDF")
        ax.set_title("Stress vs Strength PDFs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


        # Small table of summary stats
        def stats_summary(rv):
            m = rv.mean()
            v = rv.var()
            s = rv.std()
            return float(m), float(s), float(v)


        s_mu, s_sd, s_var = stats_summary(stress_rv)
        str_mu, str_sd, str_var = stats_summary(strength_rv)

        st.markdown("**Summary statistics**")
        st.write({
            "Stress": {"mean": s_mu, "std": s_sd, "var": s_var},
            "Strength": {"mean": str_mu, "std": str_sd, "var": str_var},
        })

        st.caption("Tip: Increase separation between strength and stress (means) and/or reduce variability to raise R.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.markdown("Configure the distributions and click **Compute reliability**.")
    st.caption("Weibull: shape k, scale λ · Lognormal inputs are for ln(X): μ, σ · Exponential uses mean (scale).")

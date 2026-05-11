import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.signal import convolve
import sympy as sp

# =================================================================
# 1. DEFINIÇÕES E CÁLCULOS
# =================================================================


def x1(t):
    return (t**2 - 1) * np.logical_and(t >= 1, t <= 2)


def x2(t):
    cond_rect = np.logical_and(t >= -1.5, t < 0)
    cond_exp = np.logical_and(t >= 0, t <= 3)
    return np.where(cond_rect, 2, np.where(cond_exp, 2 * np.exp(-t/2), 0))


# Q1: Energia Numérica
energia_q1, _ = quad(lambda t: (t**2 - 1)**2, 1, 2)

# Q3: Parâmetros AM
A, fm, fc = 1.5, 50, 500
t3 = np.linspace(0, 0.1, 5000)
y_am = (A + np.cos(2 * np.pi * fm * t3)) * np.cos(2 * np.pi * fc * t3)

# Q4: Convolução
dt = 0.005
t4 = np.arange(0, 5, dt)
h_t = np.exp(-2 * t4) * (t4 >= 0)
x4_t = np.logical_and(t4 >= 0, t4 <= 2).astype(float)
y_conv = convolve(x4_t, h_t) * dt
t_conv = np.linspace(0, t4[-1]*2, len(y_conv))

# Q5: EDO Simbólica
t_sym = sp.symbols('t', real=True, positive=True)
v = sp.Function('v')(t_sym)
vi = 6 * sp.exp(-3 * t_sym)
lhs = v.diff(t_sym, t_sym) + 7 * v.diff(t_sym) + 10 * v
rhs = vi.diff(t_sym) + 6 * vi
ics = {v.subs(t_sym, 0): 6, v.diff(t_sym).subs(t_sym, 0): -4}
sol_edo = sp.dsolve(lhs - rhs, v, ics=ics)
v_out = sp.lambdify(t_sym, sol_edo.rhs, 'numpy')

# =================================================================
# 2. RELATÓRIO NO TERMINAL (LIMPO DE CITAÇÕES)
# =================================================================

print("="*50)
print("RELATÓRIO DE RESPOSTAS ANALÍTICAS - UFERSA")
print("="*50)
print(f"Questão 1b/c: Energia do sinal = {energia_q1:.4f} J")
print("\nQuestão 4b: Justificativa")
print("- Causal: Sim, h(t)=0 para t<0.")
print("- Estável: Sim, h(t) é absolutamente integrável.")
print(f"\nQuestão 5c: Saída v_o(t):\n{sol_edo.rhs}")
print("="*50)

# =================================================================
# 3. PLOTAGEM COM POSICIONAMENTO MANUAL
# =================================================================

fig, axs = plt.subplots(5, 1, figsize=(12, 30))
plt.subplots_adjust(hspace=1.0, top=0.95, bottom=0.05)


def set_custom_title(ax, text):
    ax.text(0.5, 1.2, text, transform=ax.transAxes,
            fontsize=13, fontweight='bold', ha='center', va='center')


# Q1
axs[0].plot(np.linspace(0, 3, 1000), x1(
    np.linspace(0, 3, 1000)), color='blue', linewidth=2)
set_custom_title(axs[0], "Questão 1: Energia do Sinal")
axs[0].grid(True)

# Q2
t2_p = np.linspace(-6, 8, 2000)
axs[1].plot(t2_p, x2(t2_p), 'black', label='Original', linewidth=2)
axs[1].plot(t2_p, x2(t2_p + 4), '--', label='Avanço')
axs[1].plot(t2_p, x2(t2_p / 2), ':', label='Expansão')
set_custom_title(axs[1], "Questão 2: Transformações")
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Q3
axs[2].plot(t3, y_am, color='purple', linewidth=0.8)
set_custom_title(axs[2], "Questão 3: Modulação AM")
axs[2].grid(True)

# Q4
axs[3].plot(t_conv, y_conv, color='orange', linewidth=2)
axs[3].set_xlim(0, 5)
set_custom_title(axs[3], "Questão 4: Saída via Convolução")
axs[3].grid(True)

# Q5
t5_p = np.linspace(0, 3, 500)
axs[4].plot(t5_p, v_out(t5_p), color='red', linewidth=2)
set_custom_title(axs[4], "Questão 5: Resposta do Circuito")
axs[4].grid(True)

plt.show()

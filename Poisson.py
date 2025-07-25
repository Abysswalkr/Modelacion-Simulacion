import math
import random
import matplotlib.pyplot as plt

# Parámetros
lam = 3.0
p   = 0.25

def poisson_pmf(i, lam=lam):
    return math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1))

def geometric_pmf(i, p=p):
    return p * (1 - p)**i

def compute_c(max_i=200):
    c = 0.0
    for i in range(max_i + 1):
        qi = geometric_pmf(i)
        if qi > 0:
            ratio = poisson_pmf(i) / qi
            if ratio > c:
                c = ratio
    return c

c = compute_c()

def sample_geometric(p=p):
    u = random.random()
    return math.floor(math.log(u) / math.log(1 - p))

def sample_poisson_rejection(c=c):
    while True:
        y = sample_geometric()
        u = random.random()
        if u <= poisson_pmf(y) / (c * geometric_pmf(y)):
            return y

# Generamos muestras
N = 10000
samples = [sample_poisson_rejection() for _ in range(N)]

# Graficamos histograma
plt.figure()
plt.hist(samples, bins=range(0, max(samples) + 2), density=True)
plt.xlabel('k')
plt.ylabel('Frecuencia relativa')
plt.title('Histograma de Poisson(3) por rechazo–aceptación')
plt.show()

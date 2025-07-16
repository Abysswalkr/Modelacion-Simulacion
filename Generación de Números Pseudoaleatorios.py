x0 = 2
a  = 1
c  = 3
m  = 10

def generar_lcg(x_prev: int) -> int:
    return (a * x_prev + c) % m

x1 = generar_lcg(x0)
print(f"X1 = {x1}")

N = 20
X = [x0]
for _ in range(1, N):
    X.append(generar_lcg(X[-1]))

print(f"Los primeros {N} valores son:\n{X}")


import matplotlib.pyplot as plt


def listes_euler_explicite(x0, y0, tmax, N):
    time = [0]
    hare = [x0]
    lynx = [y0]
    h = tmax / N
    for k in range(N):
        time.append((k + 1) * h)
        xk1 = hare[k] + h * hare[k] * (1 - lynx[k])
        yk1 = lynx[k] + h * lynx[k] * (hare[k] - 1)
        hare.append(xk1)
        lynx.append(yk1)
    return (hare, lynx, time)


def draw_lynx(x0, y0, tmax, N):
    hare, lynx, time = listes_euler_explicite(x0, y0, tmax, N)
    plt.plot(time, lynx)
    plt.xlabel("time")
    plt.ylabel("Lynx")
    plt.title("Evolution of population of lynx")
    plt.show()


def draw_hare(x0, y0, tmax, N):
    hare, lynx, time = listes_euler_explicite(x0, y0, tmax, N)
    plt.plot(time, hare)
    plt.xlabel("time")
    plt.ylabel("hares")
    plt.title("Evolution of population of hares")
    plt.show()


def draw_both(x0, y0, tmax, N):
    hare, lynx, time = listes_euler_explicite(x0, y0, tmax, N)
    pl = plt.figure()
    plt.grid()
    plt.plot(time, hare, label=u"hare")
    plt.plot(time, lynx, label=u"lynx")
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('Lynx/hare')
    plt.title("Evolution of the two populations")
    plt.show()


def draw_lynx_hare(x0, y0, tmax, N):
    for i in range(1, 5):
        hare, lynx, time = listes_euler_explicite(x0, y0, tmax, N * i)
        plt.plot(hare, lynx, label=f"h{i}")
    plt.legend(loc='best')
    plt.xlabel("hares")
    plt.ylabel("Lynx")
    plt.title("Evolution of population of lynx as a function hares")
    plt.show()


def draw_all(x0, y0, tmax, N):
    draw_lynx(x0, y0, tmax, N)
    draw_hare(x0, y0, tmax, N)
    draw_both(x0, y0, tmax, N)
    draw_lynx_hare(x0, y0, tmax, N)
draw_all(2, 1, 120, 7000)

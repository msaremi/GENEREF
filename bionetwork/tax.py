from matplotlib import pyplot as plt
import numpy as np


def de_gross_net(max):
    brackets = [
        (0, 0.0),
        (9169, 0.0),
        (9170, 0.14),
        (14255, 0.24),
        (14256, 0.24),
        (55960, 0.42),
        (55961, 0.42),
        (265236, 0.42),
        (265327, 0.45),
        (np.inf, 0.45)
    ]

    marginal_tax = np.zeros((max,))
    net_income = np.zeros_like(marginal_tax)

    b = 0
    for i in range(max):
        if i > brackets[b + 1][0]:
            b += 1

        marginal_tax[i] = (brackets[b][1]
                           + (i - brackets[b][0]) / (brackets[b + 1][0] - brackets[b][0])
                           * (brackets[b + 1][1] - brackets[b][1]))

        if i > 0:
            net_income[i] = net_income[i - 1] + (1 - marginal_tax[i])

    return net_income


# fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(de_gross_net(300000))
# ax.set_facecolor('xkcd:salmon')
plt.grid = True
plt.show()
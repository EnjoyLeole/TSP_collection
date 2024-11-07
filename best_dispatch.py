import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithm.GA import GA


def generate_random_customer(n_customer, width=1000):
    np.random.seed(42)
    rnd = np.random
    lst = []
    for i in range(n_customer):
        lst.append([rnd.randint(0, width), rnd.randint(0, width)])
    customers = np.array(lst)
    return customers


def draw_path(best_path, iteration=None, best_record=None):
    # 加上一行因为会回到起点
    fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
    axs[0].scatter(best_path[:, 0], best_path[:, 1])
    best_path = np.vstack([best_path, best_path[0]])
    axs[0].plot(best_path[:, 0], best_path[:, 1])
    axs[0].set_title('规划结果')
    if iteration is not None:
        iterations = range(iteration)
        # best_record = model.best_record
        axs[1].plot(iterations, best_record)
        axs[1].set_title('收敛曲线')
    plt.show()


def efficiency():
    lst = []
    for i in range(30):
        nc = 10 * (i + 1)
        data = generate_random_customer(nc, width=1000)
        model = GA(num_city=data.shape[0], num_total=25, iteration=500, data=data.copy())
        path, path_len = model.run()
        print(nc, path_len / nc)
        lst += [[nc, path_len / nc]]
    
    df=pd.DataFrame(lst)
    df.to_csv('GA_efficiency.csv')
    _, ax = plt.subplots()
    ax.plot([i[0] for i in lst], [i[1] for i in lst])
    plt.show()

if __name__ == '__main__':
    efficiency()

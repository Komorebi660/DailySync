import matplotlib.pyplot as plt
#from scipy import optimize
#from scipy.interpolate import interp1d
#import numpy as np

#plt.rc('font', family='Times New Roman')


def f(x, a, b):
    return a*x + b


# more than one query in one file
def dump_data(path):
    index0_list = []
    index1_list = []
    x0 = []
    x1 = []
    with open(path, "r") as f:
        i = 0
        tmp_index0_list = []
        tmp_index1_list = []
        tmp_x0 = []
        tmp_x1 = []
        for line in f.readlines():
            str_list = line.strip().split(" ")
            length = len(str_list)

            if(str_list[length-3] != "0" and str_list[length-3] != "1"):
                # next query
                index0_list.append(tmp_index0_list)
                index1_list.append(tmp_index1_list)
                x0.append(tmp_x0)
                x1.append(tmp_x1)

                tmp_index0_list = []
                tmp_index1_list = []
                tmp_x0 = []
                tmp_x1 = []
                i = 0
            else:
                i += 1
                index_id = int(str_list[length-3])
                score = float(str_list[length-1])
                if index_id == 0:
                    tmp_index0_list.append(score)
                    tmp_x0.append(i)
                elif index_id == 1:
                    tmp_index1_list.append(score)
                    tmp_x1.append(i)
    return index0_list, index1_list, x0, x1


# draw comparison
def draw(path_before, path_after):
    index0_list_before, index1_list_before, x0_before, x1_before = dump_data(path_before)
    index0_list_after, index1_list_after, x0_after, x1_after = dump_data(path_after)

    length = len(index0_list_before)
    for i in range(length):
        plt.subplot(2, 1, 1)
        plt.title('[before] k=50, rank_function = <*> + <*>')
        plt.xlabel('steps')
        plt.ylabel('total distance (lower is better)')
        plt.xlim((0, max(x0_before[i][-1], x0_after[i][-1], x1_before[i][-1], x1_after[i][-1])+100))
        plt.scatter(x0_before[i], index0_list_before[i], label='index for v1', marker='o')
        plt.scatter(x1_before[i], index1_list_before[i], label='index for v2', marker='x')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title('[after] k=50, rank_function = <*> + <*>')
        plt.xlabel('steps')
        plt.ylabel('total score (lower is better)')
        plt.xlim((0, max(x0_before[i][-1], x0_after[i][-1], x1_before[i][-1], x1_after[i][-1])+100))
        plt.scatter(x0_after[i], index0_list_after[i], label='index for v1', marker='o')
        plt.scatter(x1_after[i], index1_list_after[i], label='index for v2', marker='x')
        plt.legend()

        plt.show()


# only one query and one file
def draw_traverse(path):
    index0_list = []
    index1_list = []
    x0 = []
    x1 = []
    with open(path, "r") as f:
        i = 0
        for line in f.readlines():
            str_list = line.strip().split(" ")
            length = len(str_list)

            if((str_list[length-3] != "0" and str_list[length-3] != "1")):
                break
            else:
                i += 1
                index_id = int(str_list[length-3])
                score = float(str_list[length-1])
                if index_id == 0:
                    index0_list.append(score)
                    x0.append(i)
                elif index_id == 1:
                    index1_list.append(score)
                    x1.append(i)

    plt.figure(figsize=(16, 9))

    plt.title('')
    plt.xlabel('steps', fontsize=30)
    plt.xticks(size=20, color='gray')
    plt.ylabel('total score', fontsize=30)
    plt.yticks([], size=20, color='gray')
    plt.ylim(0.4, 2.8)
    plt.xlim(-10, 1510)

    plt.scatter(x0, index0_list, label='entity from index1', marker='o', color='#2F7FC1', alpha=0.6, s=70)
    #a_1, b_1 = optimize.curve_fit(f, x0, index0_list)[0]
    # plt.plot(x0, f(np.array(x0), a_1, b_1), linestyle='--', color='#1f77b4')
    plt.scatter(x1, index1_list, label='entity from index2', marker='^', c='white', edgecolor='#EF7A6D', s=90)
    #a_2, b_2 = optimize.curve_fit(f, x1, index1_list)[0]
    # plt.plot(x1, f(np.array(x1), a_2, b_2), linestyle='--', color='#ff7f0e')

    ax = plt.gca()
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.legend(bbox_to_anchor=(0.76, 1.12), fontsize=20, ncol=2)
    #plt.grid(axis='y', linestyle='--', color='black')
    plt.tight_layout()
    plt.savefig("multicol_traverse.svg", format="svg")
    plt.show()


if __name__ == '__main__':
    #draw("before.txt", "after.txt")
    draw_traverse("./log.txt")

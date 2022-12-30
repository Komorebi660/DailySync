time_gt = []
result_gt = []
time_before = []
result_before = []
time_after = []
result_after = []


def dump_results(path, result_set, time_set):
    num_of_scans = []
    with open(path, "r") as f:
        f.readline()
        for i in range(100):
            _ = f.readline()    # Timing is on.
            # not ground truth file
            if ("gt" not in path):
                line = f.readline()  # num_of_scans
                num_of_scans.append(int(line.split(" ")[-1]))
            _ = f.readline()    # id | score
            _ = f.readline()    # ----+-----

            temp_result = []
            for j in range(50):
                line = f.readline()
                temp_result.append(int(line.split('|')[0]))
            result_set.append(temp_result)

            _ = f.readline()  # (50 rows)
            _ = f.readline()  # \n
            line = f.readline()  # ----+-----
            time_set.append(float(line.split(" ")[1]))
            _ = f.readline()  # Timing is off.
    return num_of_scans


def calcu_time(time_set):
    return sum(time_set) / len(time_set)


def calcu_num_of_scans(num_of_scans_set):
    return sum(num_of_scans_set) / len(num_of_scans_set)


def calcu_recall(result_set, gt_set):
    recall_set = []
    for i in range(100):
        recall = len(set(result_set[i]) & set(gt_set[i])) / len(gt_set[i])
        recall_set.append(recall)
    return recall_set


if __name__ == "__main__":
    dump_results("gt.txt", result_gt, time_gt)
    num_of_scans_before = dump_results("before.txt", result_before, time_before)
    recall_set_before = calcu_recall(result_before, result_gt)

    num_of_scans_after = dump_results("after.txt", result_after, time_after)
    recall_set_after = calcu_recall(result_after, result_gt)

    print("\tbefore\t\t\t\tafter")
    print("NumOfScans\tRecall\t\tNumOfScans\tRecall")
    for i in range(100):
        print("%d\t\t%.4f\t\t%d\t\t%.4f" % (num_of_scans_before[i], recall_set_before[i], num_of_scans_after[i], recall_set_after[i]))

    case1_num = 0
    case2_num = 0
    case3_num = 0
    case4_num = 0
    for i in range(100):
        if(num_of_scans_before[i] >= num_of_scans_after[i] and recall_set_before[i] <= recall_set_after[i]):
            case1_num += 1
        elif(num_of_scans_before[i] < num_of_scans_after[i] and recall_set_before[i] == recall_set_after[i]):
            case2_num += 1
        elif(num_of_scans_before[i] < num_of_scans_after[i] and recall_set_before[i] < recall_set_after[i]):
            case3_num += 1
        elif(num_of_scans_before[i] > num_of_scans_after[i] and recall_set_before[i] > recall_set_after[i]):
            case4_num += 1
    print("case1: %.2f, case2: %.2f, case3: %.2f, case4: %.2f" % (case1_num/100, case2_num/100, case3_num/100, case4_num/100))

    print("[before] NumberOfScans = %.2f, Latency = %.2fms, Recall = %.4f"
          % (calcu_num_of_scans(num_of_scans_before), calcu_time(time_before), sum(recall_set_before)/100))
    print("[After]  NumberOfScans = %.2f, Latency = %.2fms, Recall = %.4f"
          % (calcu_num_of_scans(num_of_scans_after), calcu_time(time_after), sum(recall_set_after)/100))

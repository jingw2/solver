# /usr/bin/python 3.6
# -*-coding:utf-8-*-

import subprocess 
import psutil
import matplotlib.pyplot as plt
import time

cmd = "python ./slot_allocation/slot_allocation_app.py --params {\"horizon\":30,\"warehouse_id\":\"65c0eb0a5c113609bbba19e5246c5ed2\",\"customer_id\":\"5df83c373bde3c002cc4b4c3\",\"pick_zones\":[\"A\"],\"storage_zones\":[\"A\"],\"end_time\":\"20200518\",\"initialize_dist_matrix\":false,\"input_path\":\"./data/input\",\"output_path\":\"./data/output\",\"strategy_type\":1,\"dist_matrix_path\":\"./data/output\"}"
process = subprocess.Popen(cmd.split(" "))

pid = process.pid
print("process id: ", pid)

def get_memory_list():
    process = psutil.Process(pid)
    memory_list = []
    while process_running(process):
        try:
            memo = process.memory_info().rss / 1024 / 1024 #MB
        except:
            break
        memory_list.append(memo)
        time.sleep(2)
    return memory_list

def process_running(process):
    try:
        memo = process.memory_info().rss / 1024 / 1024
        return True 
    except:
        return False

def plot():
    start = time.time()
    memory_list = get_memory_list()
    end = time.time()
    print("Time spent to run {}s".format(round(end-start, 2)))
    plt.plot([x for x in range(len(memory_list))], memory_list)
    plt.xlabel("record point")
    plt.ylabel("memory (MB)")
    plt.show()

if __name__ == "__main__":
    plot()

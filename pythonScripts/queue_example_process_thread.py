import time
import threading
import queue

# Running on queue_thread thread
def getter(q, queue_return):
    while True:
        print("size of q1 on getter", q.qsize())
        item = q.get()
        print('getting item ...', item)
        if item is None:
            queue_return.put(None)  # Send back results
            break
        time.sleep(0.4)
        queue_return.put(item)  # Send back results

# Running on queue_thread_process_result thread
def processResult(q):
    while True:
        item = q.get()
        print("Getting result: ", item)
        time.sleep(1)
        if item is None:
            break


# Main thread operation
def operation_on_main(q, speed):
    for i in range(10):
        time.sleep(speed)
        q.put(i)
    q.put(None)  # send None to queue when done


if __name__ == '__main__':
    q1 = queue.Queue()
    q_return_data_1 = queue.Queue()
    queue_thread = threading.Thread(target=getter, args=(q1, q_return_data_1))
    queue_thread_process_result = threading.Thread(target=processResult, args=(q_return_data_1,))
    queue_thread_process_result.daemon = True
    queue_thread.daemon = True
    queue_thread_process_result.start()
    queue_thread.start()

    print("EXAMPLE : main thread - SLOWER - than getter thread")
    print("EXAMPLE : getter thread - SLOWER - than processResult thread")
    operation_on_main(q1, 0.6)
    queue_thread.join()  # Wait for thread to finish
    queue_thread_process_result.join()

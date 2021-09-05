import time
import threading, queue

results = []
def getter(q, queue_return):
    global results
    while True:
        print("size of queue", q.qsize())
        item = q.get()
        print('getting item ...', item)
        if item is None:
            break
        time.sleep(0.4)
        results.append(42)
        queue_return.put(results) # Send back results
        
# Main thread operation
def operation_on_main(q, speed):
    for i in range(10):
        time.sleep(speed)
        q.put(i)
    q.put(None) # send None to queue when done

if __name__ == '__main__':
    q1 = queue.Queue()
    q_return_data_1 = queue.Queue()
    queue_thread_slower = threading.Thread(target=getter, args=(q1,q_return_data_1))
    queue_thread_slower.daemon = True
    queue_thread_slower.start()

    q2 = queue.Queue()
    q_return_data_2 = queue.Queue()
    queue_thread_faster = threading.Thread(target=getter, args=(q2,q_return_data_2))
    queue_thread_faster.daemon = True
    queue_thread_faster.start()

    print("EXAMPLE : main thread - FASTER - than queue thread")
    operation_on_main(q1, 0.2)
    queue_thread_slower.join() # Wait for thread to finish
    print(q_return_data_1.get()) # Get results

    print("EXAMPLE : main thread - SLOWER - than queue thread")
    operation_on_main(q2, 0.6)
    queue_thread_faster.join() # Wait for thread to finish
    print(q_return_data_2.get()) # Get results
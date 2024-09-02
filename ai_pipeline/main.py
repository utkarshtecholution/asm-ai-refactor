from queue import Queue
import threading
import time

# creating an frame queue for inserting realtime frames
frame_queue = Queue(maxsize=5)

new_item_event = threading.Event()

def consumer():
    while True:
        new_item_event.wait()  # Wait until the event is set
        while not frame_queue.empty():
            document = frame_queue.get()
            print(f"Consumed: {document}")
            time.sleep(1)
            frame_queue.task_done()
        new_item_event.clear()  # Clear the event when queue is empty

consumer_thread = threading.Thread(target=consumer, daemon=True)
consumer_thread.start()

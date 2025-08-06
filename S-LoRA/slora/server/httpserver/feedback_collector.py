from threading import Thread
from queue import SimpleQueue
from time import time, sleep
import csv

class FeedbackCollector:
    def __init__(self, datapath, flush_threshold=8, max_wait_sec=60):
        self.datapath = datapath
        self.flush_threshold = flush_threshold
        self.max_wait_sec = max_wait_sec

        self.samples = {}  # req_id -> {prompt, completion (list), label, timestamp}
        self.queue = SimpleQueue()  # event queue

        self.running = True
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit_update(self, req_id, prompt=None, completion=None, label=None):
        self.queue.put(("update", req_id, prompt, completion, label))

    def _loop(self):
        last_flush_time = time()

        while self.running:
            # Process all pending updates first
            while not self.queue.empty():
                event = self.queue.get()
                if event[0] == "update":
                    _, req_id, prompt, completion, label = event
                    sample = self.samples.get(req_id)
                    if sample is None:
                        if prompt is None:                           
                            continue  # skip invalid update
                        sample = {"prompt": prompt, "completion": [], "label": None, "timestamp": time()}
                        self.samples[req_id] = sample
                    if completion is not None:
                        sample["completion"].append(completion)
                    if label is not None:
                        sample["label"] = label
                    sample["timestamp"] = time()

            now = time()
            to_flush = []
            to_remove = []

            # Check for valid or stale samples
            for req_id, sample in list(self.samples.items()):
                age = now - sample["timestamp"]
                if sample["label"] is not None:
                    to_flush.append((req_id, sample))
                elif age > self.max_wait_sec:
                    to_remove.append(req_id)

            # Flush if needed
            if len(to_flush) >= self.flush_threshold or (to_flush and now - last_flush_time > 10):
                self._flush(to_flush)
                last_flush_time = now
                for req_id, _ in to_flush:
                    self.samples.pop(req_id, None)

            # Remove stale samples
            for req_id in to_remove:
                print(f"[FeedbackCollector] Removing stale sample: {req_id}")
                self.samples.pop(req_id, None)

            sleep(1)

    def _flush(self, items):
        print("###### Flushing feedback samples to file #####")
        try:
            with open(self.datapath, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for _, sample in items:
                    writer.writerow([
                        sample["prompt"],
                        "".join(sample["completion"]),
                        sample["label"]
                    ])
            print(f"[FeedbackCollector] Flushed {len(items)} samples.")
        except Exception as e:
            print(f"[FeedbackCollector] Flush failed: {e}")

    def stop(self):
        self.running = False
        self.thread.join()
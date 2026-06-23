import cv2
import time
import threading
from queue import Queue

class LowLatencyVideoProcessor:
    def __init__(self, stream_source=0, target_width=640, target_height=640):
        self.stream_source = stream_source
        self.target_width = target_width
        self.target_height = target_height
        
        # Thread-safe queue to hold only the latest frame
        self.frame_queue = Queue(maxsize=1)
        self.running = False

    def _frame_producer(self):
        """Thread 1: Read frames from camera at maximum speed with zero blocking"""
        cap = cv2.VideoCapture(self.stream_source)
        if not cap.isOpened():
            print("[Error] Camera stream not opened!")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Agar queue full hai, toh purana frame drop karo aur latest insert karo
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            self.frame_queue.put(frame)
            
        cap.release()

    def run_inference_pipeline(self, frame_stride=3):
        """Thread 2 (Main): Consume frames, resize, apply stride, and run dummy inference"""
        self.running = True
        
        # Start the background producer thread
        producer_thread = threading.Thread(target=self._frame_producer, daemon=True)
        producer_thread.start()

        frame_count = 0
        print("[System] Low-latency pipeline running. Press 'q' to exit.")

        while self.running:
            if not self.frame_queue.empty():
                raw_frame = self.frame_queue.get()
                frame_count += 1

                # Step 1: Immediately resize for the model (Optimization)
                resized_frame = cv2.resize(raw_frame, (self.target_width, self.target_height))

                # Step 2: Frame Skipping Logic (Stride) to save compute power
                if frame_count % frame_stride == 0:
                    # Yahan aap apna actual model call kar sakte hain: model(resized_frame)
                    # Fake processing delay representing model inference (e.g., YOLO)
                    time.sleep(0.010) # 10ms inference time simulation
                    
                    # Visual indicators for interviewers
                    cv2.putText(resized_frame, "Inference Active", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Step 3: Show the optimized frame
                cv2.imshow("Video inference low latency processing", resized_frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use 0 for local webcam, or pass a video path string
    processor = LowLatencyVideoProcessor(stream_source=0)
    processor.run_inference_pipeline(frame_stride=3)
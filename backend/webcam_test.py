"""
webcam_test.py — Minimal OpenCV Window Test
"""
import cv2
import time

def test():
    print(" [Test] Attempting to open webcam and window...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" [Error] Camera not found.")
        return

    win_name = "TEST_WINDOW"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    
    start_t = time.time()
    print(" [Test] Starting loop. Close the window or press ESC to stop.")
    
    while time.time() - start_t < 15: # 15 second test
        ret, frame = cap.read()
        if not ret: break
        
        cv2.putText(frame, "IF YOU SEE THIS, CLICK IT", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(win_name, frame)
        
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()
    print(" [Test] Finished.")

if __name__ == "__main__":
    test()

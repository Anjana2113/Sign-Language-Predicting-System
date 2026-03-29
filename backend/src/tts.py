"""
src/tts.py — Rock-Solid Thread-Safe Text-to-Speech
"""
import pyttsx3
import threading
import queue
import time
import os

class TTSEngine:
    """
    Ultra-robust TTS engine.
    Re-initializes a fresh engine for each speech request to prevent hanging.
    """
    def __init__(self, rate=150, volume=1.0):
        self.queue = queue.Queue()
        self.rate = rate
        self.volume = volume
        
        # Persistent worker thread
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self):
        """
        Background loop that processes segments from the queue.
        """
        while True:
            try:
                # Wait for text
                text = self.queue.get() 
                if not text: continue
                
                print(f" [TTS] Speaking: {text}")
                
                # FRESH INITIALIZATION PER REQUEST
                # This is the most reliable way on Windows to avoid 'Speaks only once' bug.
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                
                voices = engine.getProperty("voices")
                if len(voices) > 1:
                    engine.setProperty("voice", voices[1].id)

                engine.say(text)
                engine.runAndWait()
                
                # Cleanup
                engine.stop()
                del engine
                
                self.queue.task_done()
                print(f" [TTS] Finished speaking.")
                
            except Exception as e:
                print(f" [TTS Error] {e}")
                time.sleep(0.5)

    def speak(self, text):
        """Non-blocking speak (adds to queue)."""
        if not text.strip(): return
        # Limit the queue size to avoid 'stuttering'
        if self.queue.qsize() < 2:
            self.queue.put(text)

    def speak_sync(self, text):
        """Synchronous speak (blocks until done)."""
        if not text.strip(): return
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except: pass

if __name__ == "__main__":
    tts = TTSEngine()
    tts.speak("Test one.")
    time.sleep(3)
    tts.speak("Test two.")
    time.sleep(3)

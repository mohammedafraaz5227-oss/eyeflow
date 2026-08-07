"""Benchmark interaction performance for the Intelligent Interaction Engine."""

import time
import math
import random

class InteractionBenchmark:
    """Measures usability metrics of the eye tracking interaction pipeline."""
    
    def __init__(self):
        self.metrics = {
            'target_acquisition_times': [],
            'false_clicks': 0,
            'success_clicks': 0,
            'travel_distance': 0.0
        }
        
        self._last_x = None
        self._last_y = None
        
        self.is_running = False
        self._current_target = None
        self._target_spawn_time = 0.0

    def start_benchmark(self):
        print("Starting Interaction Benchmark...")
        print("Please click on the targets as they appear.")
        self.is_running = True
        self._spawn_target()
        
    def stop_benchmark(self):
        self.is_running = False
        print("\n--- Benchmark Results ---")
        avg_time = sum(self.metrics['target_acquisition_times']) / len(self.metrics['target_acquisition_times']) if self.metrics['target_acquisition_times'] else 0
        print(f"Average Target Acquisition Time: {avg_time:.2f} seconds")
        print(f"Successful Clicks: {self.metrics['success_clicks']}")
        print(f"False Clicks: {self.metrics['false_clicks']}")
        print(f"Total Cursor Travel Distance: {self.metrics['travel_distance']:.0f} pixels")

    def _spawn_target(self):
        # 50x50 target
        self._current_target = (random.randint(100, 1800), random.randint(100, 900))
        self._target_spawn_time = time.time()
        print(f"New target at {self._current_target}")

    def update(self, cursor_x, cursor_y, is_clicking):
        if not self.is_running:
            return
            
        # Update travel distance
        if self._last_x is not None:
            self.metrics['travel_distance'] += math.sqrt((cursor_x - self._last_x)**2 + (cursor_y - self._last_y)**2)
            
        self._last_x = cursor_x
        self._last_y = cursor_y
        
        # Check clicks
        if is_clicking:
            if self._current_target:
                tx, ty = self._current_target
                # Hitbox of 50px radius
                if math.sqrt((cursor_x - tx)**2 + (cursor_y - ty)**2) < 50:
                    acq_time = time.time() - self._target_spawn_time
                    self.metrics['target_acquisition_times'].append(acq_time)
                    self.metrics['success_clicks'] += 1
                    print(f"Target Hit! Time: {acq_time:.2f}s")
                    
                    if self.metrics['success_clicks'] >= 10:
                        self.stop_benchmark()
                    else:
                        self._spawn_target()
                else:
                    self.metrics['false_clicks'] += 1
                    print("False click off target!")

if __name__ == "__main__":
    benchmark = InteractionBenchmark()
    benchmark.start_benchmark()
    
    # Simulate a user hitting the first target after 1.5s
    time.sleep(1.5)
    benchmark.update(benchmark._current_target[0], benchmark._current_target[1], True)
    benchmark.stop_benchmark()

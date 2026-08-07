import sys
import os
import time
import logging
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.manager import ConfigManager
from core.pipeline import PipelineController
from core.types import TrackingState

class TestTrackingPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        cls.config_mgr = ConfigManager()
        cls.settings = cls.config_mgr.load()
        # Ensure we don't pause on start so we process frames immediately
        cls.settings.pause_on_start = False
        
    def test_01_startup_and_tracking(self):
        pipeline = PipelineController(self.settings)
        
        # Test INITIALIZING state
        self.assertEqual(pipeline.state, TrackingState.INITIALIZING)
        
        # Start pipeline
        pipeline.start()
        time.sleep(1.0) # wait for camera to warm up
        
        self.assertTrue(pipeline.is_running)
        self.assertEqual(pipeline.state, TrackingState.TRACKING)
        
        # Wait for a few frames to be processed
        time.sleep(2.0)
        
        # Verify fps is computing
        self.assertGreater(pipeline.fps, 0.0)
        
        # Test shutdown
        pipeline.stop()
        self.assertFalse(pipeline.is_running)
        self.assertEqual(pipeline.state, TrackingState.STOPPED)

    def test_02_pause_resume(self):
        pipeline = PipelineController(self.settings)
        pipeline.start()
        time.sleep(1.0)
        
        self.assertEqual(pipeline.state, TrackingState.TRACKING)
        pipeline.pause()
        self.assertEqual(pipeline.state, TrackingState.PAUSED)
        self.assertTrue(pipeline.is_paused)
        
        pipeline.resume()
        self.assertEqual(pipeline.state, TrackingState.TRACKING)
        self.assertFalse(pipeline.is_paused)
        
        pipeline.stop()

if __name__ == '__main__':
    unittest.main()

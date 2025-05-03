#!/usr/bin/env python3
"""
VisionBot - Efficient Vision Chatbot with Perceptual Hashing
A simple implementation of a vision-based chatbot that uses perceptual 
hashing (pHash) to reduce processing overhead and optimize resource usage.

This implementation demonstrates how to use perceptual hashing to only 
process frames that have significantly changed, reducing API costs 
and improving response time.
"""

import asyncio
import time
import logging
import cv2
import numpy as np
import argparse
import os
from PIL import Image
import imagehash
from concurrent.futures import ThreadPoolExecutor
import json
from enum import Enum
import threading
from datetime import datetime

# Uncomment and configure OpenAI integration if you have API access
# import openai
# openai.api_key = os.environ.get("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("visionbot.log")
    ]
)
logger = logging.getLogger("VisionBot")

class ScenarioType(Enum):
    """Enum defining the different types of scenarios the system can operate in"""
    STATIC = "static"
    SEMI_DYNAMIC = "semi_dynamic"
    DYNAMIC = "dynamic"

class PerformanceMetrics:
    """Track and store performance metrics for the system"""
    
    def __init__(self):
        self.hash_times = []
        self.processing_times = []
        self.skipped_frames = 0
        self.processed_frames = 0
        self.last_hash_time = 0
        self.scenario_counts = {
            ScenarioType.STATIC: 0,
            ScenarioType.SEMI_DYNAMIC: 0,
            ScenarioType.DYNAMIC: 0
        }
        
    def add_hash_time(self, time_ms):
        self.hash_times.append(time_ms)
        self.last_hash_time = time_ms
        
    def add_processing_time(self, time_ms):
        self.processing_times.append(time_ms)
        
    def increment_processed(self):
        self.processed_frames += 1
        
    def increment_skipped(self):
        self.skipped_frames += 1
        
    def increment_scenario(self, scenario_type):
        self.scenario_counts[scenario_type] += 1
        
    def get_summary(self):
        if not self.hash_times:
            return {"error": "No metrics collected yet"}
            
        return {
            "avg_hash_time_ms": sum(self.hash_times) / len(self.hash_times) if self.hash_times else 0,
            "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            "total_frames": self.processed_frames + self.skipped_frames,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "skip_rate": self.skipped_frames / (self.processed_frames + self.skipped_frames) if (self.processed_frames + self.skipped_frames) > 0 else 0,
            "scenario_distribution": {k.value: v for k, v in self.scenario_counts.items()}
        }
        
    def log_metrics(self):
        summary = self.get_summary()
        logger.info(f"Performance Summary: {json.dumps(summary, indent=2)}")

class ScenarioDetector:
    """Detect the current scenario based on frame characteristics"""
    
    def __init__(self):
        self.prev_frame = None
        self.motion_history = []
        self.edge_history = []
        self.history_size = 10  # Store last 10 frames worth of data
        
    def detect(self, frame):
        """
        Detect scenario type from frame characteristics
        Args:
            frame (np.ndarray): The current video frame
            
        Returns:
            ScenarioType: The detected scenario type
        """
        # Convert to grayscale
        if frame is None or frame.size == 0:
            return ScenarioType.STATIC
            
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        self.edge_history.append(edge_density)
        
        # Calculate motion
        motion_magnitude = 0
        if self.prev_frame is not None:
            # Simple frame difference for motion estimation
            frame_diff = cv2.absdiff(gray, self.prev_frame)
            motion_magnitude = np.mean(frame_diff) / 255.0
            self.motion_history.append(motion_magnitude)
            
        self.prev_frame = gray.copy()
        
        # Keep history at specified size
        if len(self.motion_history) > self.history_size:
            self.motion_history.pop(0)
        if len(self.edge_history) > self.history_size:
            self.edge_history.pop(0)
            
        # Calculate average motion and edge metrics
        avg_motion = sum(self.motion_history) / len(self.motion_history) if self.motion_history else 0
        avg_edge = sum(self.edge_history) / len(self.edge_history) if self.edge_history else 0
        
        # Classify scenario based on motion and edge metrics
        if avg_motion < 0.01 and avg_edge < 0.1:
            return ScenarioType.STATIC
        elif avg_motion < 0.05:
            return ScenarioType.SEMI_DYNAMIC
        else:
            return ScenarioType.DYNAMIC
  
class MemoryTracker:
    """Track memory usage of the hash system"""
    
    def __init__(self, max_entries=1000):
        self.hashes = {}  # timestamp -> hash
        self.max_entries = max_entries
        self.total_memory = 0
        self.hash_size = 8  # bytes per hash
        
    def add_hash(self, image_hash, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
            
        # If we've reached max entries, remove oldest
        if len(self.hashes) >= self.max_entries:
            oldest_key = min(self.hashes.keys())
            del self.hashes[oldest_key]
            
        self.hashes[timestamp] = image_hash
        self.update_memory_usage()
        
    def cleanup_old_hashes(self, max_age_seconds=3600):
        """Remove hashes older than max_age_seconds"""
        current_time = time.time()
        old_timestamps = [t for t in self.hashes.keys() if current_time - t > max_age_seconds]
        
        for timestamp in old_timestamps:
            del self.hashes[timestamp]
            
        self.update_memory_usage()
        
    def update_memory_usage(self):
        """Update the total memory usage calculation"""
        num_hashes = len(self.hashes)
        hash_memory = num_hashes * self.hash_size
        metadata_memory = num_hashes * 16  # timestamp (8) + dictionary overhead
        self.total_memory = hash_memory + metadata_memory
        
    def get_current_usage(self):
        """Get current memory usage in bytes"""
        return self.total_memory

class VisionBot:
    """
    Main vision chatbot implementation using perceptual hashing for optimization
    """
    
    def __init__(self, config=None):
        """
        Initialize the vision chatbot
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.memory_tracker = MemoryTracker(max_entries=self.config['max_hash_entries'])
        self.scenario_detector = ScenarioDetector()
        self.metrics = PerformanceMetrics()
        
        # State variables
        self.last_processed_hash = None
        self.last_processed_time = 0
        self.last_scenario = None
        self.running = False
        self.last_response = None
        
        # Processing thread
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Mock AI response - in production, replace with actual API call
        self.mock_responses = [
            "I can see a person sitting at a desk.",
            "There appears to be a computer monitor visible.",
            "I notice several books on a shelf in the background.",
            "There's a window showing some trees outside.",
            "The lighting in this room appears to be artificial.",
            "There's a coffee mug on the desk.",
            "I can see what looks like a keyboard and mouse.",
            "There appears to be a plant in the corner of the room.",
            "The walls seem to be painted in a light color.",
            "There's a chair visible in the frame."
        ]
        
        logger.info(f"VisionBot initialized with config: {json.dumps(self.config, indent=2)}")
        
    def _get_default_config(self):
        """Get default configuration values"""
        return {
            'thresholds': {
                ScenarioType.STATIC.value: 0.20,
                ScenarioType.SEMI_DYNAMIC.value: 0.15,
                ScenarioType.DYNAMIC.value: 0.10
            },
            'max_age_seconds': {
                ScenarioType.STATIC.value: 120,
                ScenarioType.SEMI_DYNAMIC.value: 30,
                ScenarioType.DYNAMIC.value: 10
            },
            'max_hash_entries': 1000,
            'debug_mode': True,
            'show_visualization': True,
            'visualize_hash': True
        }
        
    def compute_phash(self, frame):
        """
        Compute perceptual hash for a frame
        
        Args:
            frame (np.ndarray): The frame to compute hash for
            
        Returns:
            imagehash.ImageHash: The computed perceptual hash
        """
        start_time = time.time()
        
        # Convert OpenCV image to PIL
        if frame is None:
            return None
            
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Compute perceptual hash
        img_hash = imagehash.phash(pil_img)
        
        # Record metrics
        hash_time_ms = (time.time() - start_time) * 1000
        self.metrics.add_hash_time(hash_time_ms)
        
        # Add to memory tracker
        self.memory_tracker.add_hash(img_hash)
        
        return img_hash
        
    def calculate_similarity(self, hash1, hash2):
        """
        Calculate similarity between two perceptual hashes
        
        Args:
            hash1 (imagehash.ImageHash): First hash
            hash2 (imagehash.ImageHash): Second hash
            
        Returns:
            float: Similarity between hashes (1.0 = identical, 0.0 = completely different)
        """
        if hash1 is None or hash2 is None:
            return 0.0
            
        max_hash_diff = len(hash1.hash.flatten()) # Maximum possible difference
        actual_diff = hash1 - hash2
        
        # Convert hash difference to similarity score (1.0 = identical, 0.0 = completely different)
        similarity = 1.0 - (actual_diff / max_hash_diff)
        return similarity
        
    def should_process_frame(self, current_hash, scenario_type):
        """
        Determine if a frame should be processed based on hash comparison
        
        Args:
            current_hash (imagehash.ImageHash): Hash of current frame
            scenario_type (ScenarioType): Current scenario type
            
        Returns:
            bool: True if frame should be processed, False otherwise
        """
        # If this is the first frame, always process it
        if self.last_processed_hash is None:
            return True
            
        # Get threshold for current scenario
        threshold = self.config['thresholds'][scenario_type.value]
        
        # Calculate similarity and check against threshold
        similarity = self.calculate_similarity(current_hash, self.last_processed_hash)
        change_percent = 1.0 - similarity
        
        # Check if change exceeds threshold
        should_process = change_percent > threshold
        
        # Also check time-based processing - process at least one frame every max_age_seconds
        if not should_process:
            max_age = self.config['max_age_seconds'][scenario_type.value]
            time_since_last = time.time() - self.last_processed_time
            if time_since_last > max_age:
                should_process = True
                logger.debug(f"Processing due to age: {time_since_last:.1f}s > {max_age}s")
                
        return should_process
        
    async def process_frame(self, frame):
        """
        Process a single video frame
        
        Args:
            frame (np.ndarray): The video frame to process
            
        Returns:
            dict: Processing result information
        """
        start_time = time.time()
        
        # Detect scenario
        scenario_type = self.scenario_detector.detect(frame)
        self.metrics.increment_scenario(scenario_type)
        self.last_scenario = scenario_type
        
        # Compute hash
        current_hash = self.compute_phash(frame)
        
        # Determine if frame should be processed
        should_process = self.should_process_frame(current_hash, scenario_type)
        
        # Record metrics
        if should_process:
            self.metrics.increment_processed()
            self.last_processed_hash = current_hash
            self.last_processed_time = time.time()
            
            # In a real implementation, this would call the vision API
            # For this example, we'll mock the response
            self.last_response = await self._mock_vision_api_call(frame)
        else:
            self.metrics.increment_skipped()
            
        # Calculate processing time
        process_time_ms = (time.time() - start_time) * 1000
        self.metrics.add_processing_time(process_time_ms)
        
        return {
            'should_process': should_process,
            'scenario': scenario_type.value,
            'process_time_ms': process_time_ms,
            'response': self.last_response if should_process else self.last_response,
            'timestamp': datetime.now().isoformat()
        }
        
    async def _mock_vision_api_call(self, frame):
        """
        Mock a vision API call - in production, replace with actual API call
        
        Args:
            frame (np.ndarray): The frame to analyze
            
        Returns:
            str: Response text
        """
        # Simulate API latency (0.5-1.5 seconds)
        await asyncio.sleep(0.5 + (np.random.random() * 1.0))
        
        # Select random response from mock responses
        response = np.random.choice(self.mock_responses)
        
        # In production, this would be:
        # response = await openai.ChatCompletion.acreate(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": "You are an AI assistant that can see and describe images."},
        #         {"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64_image}}]}
        #     ]
        # )
        
        return response
        
    def create_visualization(self, frame, result):
        """
        Create visualization frame with debug information
        
        Args:
            frame (np.ndarray): Original video frame
            result (dict): Processing result information
            
        Returns:
            np.ndarray: Visualization frame
        """
        if frame is None:
            return None
            
        # Create a copy of the frame
        viz_frame = frame.copy()
        
        # Add scenario information
        cv2.putText(
            viz_frame,
            f"Scenario: {result['scenario']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add processing information
        status_color = (0, 255, 0) if result['should_process'] else (0, 0, 255)
        status_text = "PROCESSED" if result['should_process'] else "SKIPPED"
        cv2.putText(
            viz_frame,
            status_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )
        
        # Add performance metrics
        cv2.putText(
            viz_frame,
            f"Time: {result['process_time_ms']:.1f} ms",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        metrics = self.metrics.get_summary()
        skip_rate = metrics.get('skip_rate', 0) * 100
        cv2.putText(
            viz_frame,
            f"Skip Rate: {skip_rate:.1f}%",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Add response
        if self.last_response:
            # Draw semi-transparent box for text background
            response_height = 100
            overlay = viz_frame.copy()
            cv2.rectangle(
                overlay,
                (0, viz_frame.shape[0] - response_height),
                (viz_frame.shape[1], viz_frame.shape[0]),
                (0, 0, 0),
                -1
            )
            
            # Add response text
            cv2.putText(
                overlay,
                f"AI: {self.last_response}",
                (10, viz_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Add overlay with transparency
            cv2.addWeighted(overlay, 0.7, viz_frame, 0.3, 0, viz_frame)
            
        # If hash visualization is enabled, display hash
        if self.config['visualize_hash'] and self.last_processed_hash is not None:
            # Visualize the hash as a 8x8 binary grid (hash is 64 bits)
            hash_size = 8
            cell_size = 20
            offset_x = viz_frame.shape[1] - (hash_size * cell_size) - 10
            offset_y = 10
            
            # Draw border around hash
            cv2.rectangle(
                viz_frame,
                (offset_x - 5, offset_y - 5),
                (offset_x + (hash_size * cell_size) + 5, offset_y + (hash_size * cell_size) + 5),
                (255, 255, 255),
                2
            )
            
            # Flatten hash bits and draw cells
            hash_bits = self.last_processed_hash.hash.flatten()
            for i in range(hash_size * hash_size):
                row = i // hash_size
                col = i % hash_size
                
                x = offset_x + (col * cell_size)
                y = offset_y + (row * cell_size)
                
                cell_color = (0, 0, 255) if hash_bits[i] else (255, 0, 0)
                cv2.rectangle(
                    viz_frame,
                    (x, y),
                    (x + cell_size, y + cell_size),
                    cell_color,
                    -1
                )
                
        return viz_frame
        
    async def run_webcam(self, camera_id=0):
        """
        Run the vision chatbot using a webcam source
        
        Args:
            camera_id (int): Webcam ID to use
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open webcam {camera_id}")
            return
            
        logger.info(f"Started webcam session with camera {camera_id}")
        self.running = True
        
        try:
            while self.running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from webcam")
                    continue
                    
                # Process frame
                result = await self.process_frame(frame)
                
                # Create visualization
                if self.config['show_visualization']:
                    viz_frame = self.create_visualization(frame, result)
                    cv2.imshow("VisionBot", viz_frame)
                    
                # Break loop on 'q' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Stopping...")
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            self.metrics.log_metrics()
            logger.info("Webcam session ended")
            
    async def run_video(self, video_path):
        """
        Run the vision chatbot on a video file
        
        Args:
            video_path (str): Path to video file
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file {video_path}")
            return
            
        logger.info(f"Started video session with file {video_path}")
        self.running = True
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        try:
            frame_index = 0
            while self.running and frame_index < frame_count:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                result = await self.process_frame(frame)
                
                # Create visualization
                if self.config['show_visualization']:
                    viz_frame = self.create_visualization(frame, result)
                    cv2.imshow("VisionBot", viz_frame)
                    
                # Break loop on 'q' key
                key = cv2.waitKey(int(1000/fps)) & 0xFF
                if key == ord('q'):
                    break
                    
                frame_index += 1
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Stopping...")
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            self.metrics.log_metrics()
            logger.info("Video session ended")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="VisionBot - Efficient Vision Chatbot using Perceptual Hashing")
    parser.add_argument("--source", type=str, default="webcam", choices=["webcam", "video"],
                        help="Source type (webcam or video)")
    parser.add_argument("--input", type=str, default="0",
                        help="Source input (camera ID for webcam, file path for video)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Show visualization window")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode with additional logging")
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_args()
    
    # Set up configuration
    config = {
        'thresholds': {
            ScenarioType.STATIC.value: 0.20,
            ScenarioType.SEMI_DYNAMIC.value: 0.15,
            ScenarioType.DYNAMIC.value: 0.10
        },
        'max_age_seconds': {
            ScenarioType.STATIC.value: 120,
            ScenarioType.SEMI_DYNAMIC.value: 30,
            ScenarioType.DYNAMIC.value: 10
        },
        'max_hash_entries': 1000,
        'debug_mode': args.debug,
        'show_visualization': args.visualize,
        'visualize_hash': True
    }
    
    # Initialize chatbot
    chatbot = VisionBot(config)
    
    # Run appropriate source
    if args.source == "webcam":
        camera_id = int(args.input) if args.input.isdigit() else 0
        await chatbot.run_webcam(camera_id)
    elif args.source == "video":
        await chatbot.run_video(args.input)
    else:
        logger.error(f"Unknown source type: {args.source}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}") 
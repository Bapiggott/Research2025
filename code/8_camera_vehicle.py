#!/usr/bin/env python
"""
| File: 8_camera_vehicle.py
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API, 
| where the data is send/received through mavlink, the vehicle is controled using mavlink and
| camera data is sent to ROS2 topics at the same time.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp
import os

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.graphical_sensors.lidar import Lidar
from pegasus.simulator.logic.backends.mavlink_backend import MavlinkBackend, MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation
import requests
import cv2
import numpy as np
import time
import json
import asyncio
import threading


class PegasusApp:
    def __init__(self):
        # Initialization steps
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.processing = False
        world_choice = "b"
        if world_choice == "a":
            self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

            from omni.isaac.core.objects import DynamicCuboid
            import numpy as np
            cube_2 = self.world.scene.add(
                DynamicCuboid(
                    prim_path="/new_cube_2",
                    name="cube_1",
                    position=np.array([-3.0, 0, 2.0]),
                    scale=np.array([1.0, 1.0, 1.0]),
                    size=1.0,
                    color=np.array([255, 0, 0]),
                )
            )
        else:
            self.pg.load_environment(SIMULATION_ENVIRONMENTS[
                                         "Hospital"])  # "/home/liu/Downloads/USD/test1.usd") # SIMULATION_ENVIRONMENTS["Office"]) # Curved Gridroom"])

        # Configure multirotor with camera
        config_multirotor = MultirotorConfig()
        mavlink_config = MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": "/home/liu/PX4-Autopilot"
        })
        config_multirotor.backends = [
            MavlinkBackend(mavlink_config),
            ROS2Backend(vehicle_id=1, config={
                "namespace": 'drone',
                "pub_sensors": False,
                "pub_graphical_sensors": True,
                "pub_state": True,
                "sub_control": False
            })
        ]
        self.camera = MonocularCamera("camera", config={"update_rate": 60.0})
        config_multirotor.graphical_sensors = [self.camera]

        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        self.world.reset()
        self.stop_sim = False

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """
        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

            # Process camera data and interact with YOLO server
            #self.process_camera_data()
            if not self.processing:
                threading.Thread(target=self.send_camera_data).start()

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

    '''async def simulation_loop(self):
        """Main simulation loop."""
        while simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
            await asyncio.sleep(0)  # Yield control to allow camera processing

    async def process_camera_data_periodically(self):
        """
        Asynchronously processes camera data at a lower frequency (e.g., every 2 seconds).
        """
        while not self.stop_sim:
            self.process_camera_data()
            await asyncio.sleep(2.0)  # Adjust frequency as desired

    def process_camera_data(self):
        """
        Method to process camera data and interact with the YOLO server.
        """
        # YOLO server URL for detection
        yolo_detect_url = "http://localhost:5000/detect"

        # Fetch image data from the camera
        state = self.camera.state
        if state and "camera" in state:
            img_data = state["camera"].get_rgba()

            # Convert RGBA image data to BGR format expected by OpenCV
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)

            # Get the current timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_filename = f"image_{timestamp}.jpg"
            img_path = os.path.join("images", img_filename)  # Ensure "images" directory exists

            # Save the image locally
            cv2.imwrite(img_path, img_bgr)

            # Encode image as JPEG
            success, img_encoded = cv2.imencode('.jpg', img_bgr)
            if not success:
                print("Failed to encode image")
                return

            img_bytes = img_encoded.tobytes()

            # Send the image to the YOLO server for detection
            response = requests.post(yolo_detect_url, files={'image': img_bytes})

            # Process the detection results
            if response.status_code == 200:
                detection_results = response.json()
                print(f"Response: {response.text}")
                if detection_results:
                    print("Immediate Detections:", detection_results)
            else:
                print(f"Failed to get response from YOLO server: {response.status_code}")
                print(f"Error content: {response.text}")
                return

        #time.sleep(2.0)'''

    def send_camera_data(self):
        """Send image data to the YOLO server without blocking main loop."""
        self.processing = True  # Set flag to indicate processing started
        try:
            # Fetch image data
            state = self.camera.state
            if state and "camera" in state:
                img_data = state["camera"].get_rgba()
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)

                # Encode image as JPEG
                success, img_encoded = cv2.imencode('.jpg', img_bgr)
                if not success:
                    print("Failed to encode image")
                    self.processing = False
                    return

                img_bytes = img_encoded.tobytes()
                yolo_detect_url = "http://localhost:5000/detect"

                # Send image to YOLO server
                response = requests.post(yolo_detect_url, files={'image': img_bytes})

                # Optional: Handle response if needed, e.g., log if status isn't 200
                if response.status_code != 200:
                    print(f"Failed with status: {response.status_code}")

        except Exception as e:
            print(f"Error in send_camera_data: {e}")

        self.processing = False

    '''def get_recent_detections(self):
        """
        Retrieves the last 5 detections from the YOLO server.
        """
        yolo_recent_detections_url = "http://localhost:5000/detections/recent"
        try:
            response = requests.get(yolo_recent_detections_url)
            if response.status_code == 200:
                recent_detections = response.json()
                return recent_detections
            elif response.status_code == 404:
                print("No recent detection results available.")
                return []
            else:
                print(f"Error: Received status code {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return []'''


import asyncio


def main():
    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()

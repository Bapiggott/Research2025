import asyncio
import math

#from markdown_it.rules_block import heading
from mavsdk import System
import requests
from datetime import datetime
from mavsdk.offboard import OffboardError, PositionNedYaw
import requests
from mavsdk.offboard import VelocityNedYaw
#from functions import changing_elevation
import datetime
import os
import shutil

drone = System()
starting_position = None

class Position:
    def __init__(self, latitude_deg=None, longitude_deg=None, absolute_altitude_m=None, heading=None):
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.absolute_altitude_m = absolute_altitude_m
        self.heading = heading

starting_position = Position()

# Function to ensure timeout with telemetry checks
async def timeout_check(timeout_seconds, check_function):
    start_time = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
        if await check_function():
            return True
        await asyncio.sleep(1)  # Avoid tight loops
    print("Timeout reached.")
    return False

# Helper function to calculate distance between two GPS coordinates
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

# Connect to the drone
async def connect_drone():
    global drone
    try:
        await drone.connect(system_address="udp://:14540")
    except Exception as e:
        print(f"Error connecting to the drone: {e}")
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected")
            break

# Ensure the drone is armed and has taken off (without offboard)
async def ensure_armed_and_taken_off():
    global drone

    is_armed = False
    async for armed in drone.telemetry.armed():
        is_armed = armed
        break

    if not is_armed:
        print("Arming drone...")
        try:
            await drone.action.arm()
            print("Drone armed")
        except Exception as e:
            print(f"Failed to arm the drone: {e}")
            return
    else:
        print("Drone is already armed")

    in_air = False
    async for air in drone.telemetry.in_air():
        in_air = air
        break

    if not in_air:
        print("Taking off...")
        try:
            async for position in drone.telemetry.position():
                starting_altitude = position.relative_altitude_m
                print(f"Current altitude: {starting_altitude} meters")
                break

            await drone.action.set_takeoff_altitude(1.5)
            await drone.action.takeoff()

            # Use a timeout to wait until the drone reaches the target altitude
            await timeout_check(2000, lambda: asyncio.ensure_future(reached_target_altitude(1.5, starting_altitude)))
            async for position in drone.telemetry.position():
                starting_position.latitude_deg = position.latitude_deg
                starting_position.longitude_deg = position.longitude_deg
                starting_position.absolute_altitude_m = position.absolute_altitude_m
                break
            starting_position.heading = await get_heading()
        except Exception as e:
            print(f"Failed to take off: {e}")
            return
    else:
        print("Drone is already in the air")
        async for position in drone.telemetry.position():
            starting_position.latitude_deg = position.latitude_deg
            starting_position.longitude_deg = position.longitude_deg
            starting_position.absolute_altitude_m = position.absolute_altitude_m
            break
        starting_position.heading = await get_heading()

async def get_current_altitude():
    global drone
    async for position in drone.telemetry.position():
        current_altitude = position.relative_altitude_m
        #print(f"Current altitude: {current_altitude} meters")
        return current_altitude


# Check if the drone has reached a target altitude
async def reached_target_altitude(target_altitude, starting_altitude,direction ="up", tolerance=0.009, stability_count=15):
    stable_readings = 0  # Counter for how many consecutive stable readings we get
    previous_altitude = None

    async for position in drone.telemetry.position():
        #print(f"Starting alt: {starting_altitude}")
        current_altitude = await get_current_altitude() #position.relative_altitude_m
        #print(f"Current altitude: {current_altitude} meters")

        # Check if the current altitude is within the tolerance of the target altitude
        if direction  == "up":
            if current_altitude >= (target_altitude - tolerance):
                print(f"Reached target altitude of {target_altitude} meters (within tolerance of {tolerance} meters)")
                return True

            # If this is the first reading, set previous_altitude to current
            if previous_altitude is None:
                previous_altitude = current_altitude

            # Check if the altitude has been stable (within tolerance of previous altitude)
            if abs(current_altitude - previous_altitude) < tolerance:
                if current_altitude >= (target_altitude / 2.2):
                    stable_readings += 1
            else:
                stable_readings = 0  # Reset counter if the altitude fluctuates
            previous_altitude = current_altitude

            # If we've had 10 consecutive stable readings, consider the altitude stable
            if stable_readings >= stability_count:
                print(f"Altitude has been stable for {stability_count} consecutive readings.")
                return True

            # Wait a bit before the next telemetry check to avoid tight looping
            await asyncio.sleep(1)
        else:
            if current_altitude <= (target_altitude - tolerance):
                print(f"Reached target altitude of {target_altitude} meters (within tolerance of {tolerance} meters)")
                return True

            # If this is the first reading, set previous_altitude to current
            if previous_altitude is None:
                previous_altitude = current_altitude

            # Check if the altitude has been stable (within tolerance of previous altitude)
            if abs(current_altitude - previous_altitude) < tolerance:
                if current_altitude <= (target_altitude / 2.2):
                    stable_readings += 1
            else:
                stable_readings = 0  # Reset counter if the altitude fluctuates
            previous_altitude = current_altitude

            # If we've had 10 consecutive stable readings, consider the altitude stable
            if stable_readings >= stability_count:
                print(f"Altitude has been stable for {stability_count} consecutive readings.")
                return True

            # Wait a bit before the next telemetry check to avoid tight looping
            await asyncio.sleep(1)


    return False


# Return to the start position using simple location commands
async def return_to_start_position():
    global drone, starting_position

    if starting_position is None:
        print("Starting position not recorded.")
        return

    print("Returning to starting position...")
    await drone.action.goto_location(
        starting_position.latitude_deg,
        starting_position.longitude_deg,
        starting_position.absolute_altitude_m,
        starting_position.heading
    )

    # Wait for the drone to reach the target location by monitoring telemetry data
    await timeout_check(60, lambda: asyncio.ensure_future(reached_target_position(
        starting_position.latitude_deg, starting_position.longitude_deg)))

# Check if the drone has reached the desired position
async def reached_target_position(target_latitude, target_longitude, tolerance=0.2, stability_count=15):
    stable_readings = 0  # Counter for how many consecutive stable readings we get
    previous_distance = None
    async for position in drone.telemetry.position():
        current_distance = calculate_distance(position.latitude_deg, position.longitude_deg, target_latitude, target_longitude)
        #print(distance)
        """if current_distance < tolerance:  # Set a threshold for how close it should be
            print("Reached the target position.")
            return True"""
        if previous_distance is None:
            previous_distance = current_distance

        # Check if the altitude has been stable (within tolerance of previous altitude)
        if previous_distance ==  current_distance:
            if current_distance < tolerance:
                stable_readings += 1
        else:
            stable_readings = 0  # Reset counter if the altitude fluctuates
        previous_distance = current_distance

        # If we've had 10 consecutive stable readings, consider the altitude stable
        if stable_readings >= stability_count:
            print(f"Reached the target position for {stability_count} consecutive readings.")
            return True

    return False

# Move drone by setting GPS coordinates based on current position and direction
async def move_in_direction(distance: float, direction: str):
    global drone

    async for position in drone.telemetry.position():
        current_lat = position.latitude_deg
        current_lon = position.longitude_deg
        break

    heading_deg = await get_heading()
    heading_rad = math.radians(heading_deg)

    # Calculate new position based on direction and current heading
    if direction == "forward":
        new_lat = current_lat + (distance * math.cos(heading_rad)) / 111320  # Convert meters to degrees
        new_lon = current_lon + (distance * math.sin(heading_rad)) / (111320 * math.cos(math.radians(current_lat)))
    elif direction == "backward":
        new_lat = current_lat - (distance * math.cos(heading_rad)) / 111320
        new_lon = current_lon - (distance * math.sin(heading_rad)) / (111320 * math.cos(math.radians(current_lat)))
    elif direction == "left":
        new_lat = current_lat + (distance * math.cos(heading_rad + math.pi / 2)) / 111320
        new_lon = current_lon + (distance * math.sin(heading_rad + math.pi / 2)) / (111320 * math.cos(math.radians(current_lat)))
    elif direction == "right":
        new_lat = current_lat + (distance * math.cos(heading_rad - math.pi / 2)) / 111320
        new_lon = current_lon + (distance * math.sin(heading_rad - math.pi / 2)) / (111320 * math.cos(math.radians(current_lat)))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    print(f"Moving {direction} by {distance} meters...")
    await drone.action.goto_location(new_lat, new_lon, position.absolute_altitude_m, heading_deg)

    # Wait for the drone to reach the new position
    await timeout_check(2000, lambda: asyncio.ensure_future(reached_target_position(new_lat, new_lon)))

# Get the current heading
async def get_heading():
    global drone
    async for position in drone.telemetry.heading():
        print(f"Current Heading: {position.heading_deg} degrees")
        return position.heading_deg


async def reached_target_heading(target_heading, tolerance=0.01, stability_count=15):
    stable_readings = 0  # Counter for how many consecutive stable readings we get
    previous_heading = None
    async for position in drone.telemetry.position():
        current_heading = await get_heading()
        """if current_distance < tolerance:  # Set a threshold for how close it should be
            print("Reached the target position.")
            return True"""
        if previous_heading is None:
            previous_heading = current_heading

        # Check if the altitude has been stable (within tolerance of previous altitude)
        print(f'previous_heading == current_heading: {previous_heading} == {current_heading}')
        if previous_heading == current_heading:
            if abs(current_heading - target_heading) < tolerance:
                stable_readings += 1
        else:
            stable_readings = 0  # Reset counter if the altitude fluctuates
        previous_heading = current_heading

        # If we've had 10 consecutive stable readings, consider the altitude stable
        if stable_readings >= stability_count:
            print(f"Reached the target heading for {stability_count} consecutive readings.")
            return True

    return False


"""# Turn the drone by adjusting yaw
async def turn_cw(degrees: float):
    async for position in drone.telemetry.position():
        current_lat = position.latitude_deg
        current_lon = position.longitude_deg
        break
    current_altitude = await get_current_altitude()
    current_heading = await get_heading()
    new_heading = (current_heading + degrees) % 360
    #print(f"current_altitude: {current_altitude}\n")
    print(f"Turning clockwise to {new_heading} degrees")
    #await drone.action.set_yaw(new_heading)
    #await drone.action.goto_location(current_lat, current_lon, current_altitude, new_heading)
    await drone.action.goto_location(current_lat, current_lon, absolute_altitude_m=current_altitude,
                                     yaw_deg=new_heading)
    await timeout_check(2000, lambda: asyncio.ensure_future(reached_target_heading(new_heading)))

async def turn_ccw(degrees: float):
    async for position in drone.telemetry.position():
        current_lat = position.latitude_deg
        current_lon = position.longitude_deg
        break
    current_altitude = await get_current_altitude()
    current_heading = await get_heading()
    new_heading = (current_heading - degrees) % 360
    print(f"Turning counterclockwise to {new_heading} degrees")
    #await drone.action.
    await drone.action.goto_location(current_lat, current_lon, absolute_altitude_m=current_altitude, yaw_deg=new_heading)
    await timeout_check(2000, lambda: asyncio.ensure_future(reached_target_heading(new_heading)))
"""
# Function to check if the object is visible
def is_visible(object_name: str) -> bool:
    object_name = object_name.lower()
    scene_description = get_latest_detection()
    for item in scene_description:
        if object_name in item["name"].lower():
            return True
    return False

# Get object X position based on detection
def object_x(object: str) -> float:
    scene_description = get_latest_detection()
    for item in scene_description:
        if object in item["name"]:
            return ((item["xmax"] + item["xmin"]) / 2) / 1920
    return -1

# Get latest detection (from YOLO server)
def get_latest_detection():
    yolo_server_url = "http://localhost:5000/detections/latest"
    try:
        response = requests.get(yolo_server_url)
        if response.status_code == 200:
            detection_result = response.json()
            return detection_result
        elif response.status_code == 404:
            print("No detection results available.")
            return None
        else:
            print(f"Error: Received status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Function to land the drone
async def land_drone():
    global drone
    print("Landing the drone...")
    await drone.action.land()
    #await disconnect_drone()

#########

async def goto(object_name: str) -> None:
    success = await orienting(object_name)
    if success:
        await approach()
    else:
        print(f"Could not orient towards {object_name}.")


async def scan_abstract(question: str) -> bool:
    for _ in range(8):
        _1 = probe(question)
        if _1 != False:
            return _1
        await turn_cw(45)
    return False


async def scan(object_name: str) -> bool:
    for _ in range(8):
        if is_visible(object_name):
            return True
        await turn_cw(45)
    return False


async def approach() -> None:
    await move_forward(0.5)


async def orienting(object_name: str) -> bool:
    for _ in range(4):
        _1 =  object_x(object_name)
        if _1 > 0.6:
            await turn_cw(15)
        if _1 < 0.4:
            await turn_ccw(15)
        _2 = object_x(object_name)
        if 0.4 < _2 < 0.6:
            return True
    return False

def probe(question: str) -> bool:
    # Placeholder function; implement as needed
    return False


def display_image(filename):
    """
    Displays the image in the terminal using ASCII characters (optional).
    Note: This requires the 'PIL' and 'numpy' libraries.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(filename)
        img.thumbnail((80, 40))  # Resize for terminal display
        img_array = np.array(img)

        # Convert to grayscale
        img_gray = img.convert('L')
        img_array = np.array(img_gray)

        # Define ASCII characters
        ascii_chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
        ascii_str = ''
        for pixel_row in img_array:
            for pixel in pixel_row:
                ascii_str += ascii_chars[pixel // 25]
            ascii_str += '\n'

        print(ascii_str)
    except ImportError:
        print("PIL and numpy are required for displaying the image in the terminal.")
        print("You can install them using 'pip install Pillow numpy'.")

def take_picture() -> None:
    filename = get_latest_image()
    if filename:
        # Optionally display the image in the terminal
        display_image(filename)


def get_latest_image():
    """
    Retrieves the latest image from the YOLO server and saves it with a timestamp.
    """
    yolo_server_url = "http://localhost:5000/image/latest"
    try:
        response = requests.get(yolo_server_url)
        if response.status_code == 200:
            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"latest_image_{timestamp}.jpg"

            # Save the image to the current directory
            with open(filename, 'wb') as f:
                f.write(response.content)

            print(f"Image saved as {filename}")
            return filename
        else:
            print(f"Failed to get image: Status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def object_width(object_name: str) -> float:
    for _ in range(4):
        scene_description = get_latest_detection()
        if scene_description:
            for item in scene_description:
                if object_name.lower() in item["name"].lower():
                    object_wid = (item["xmax"] - item["xmin"]) / 1920
                    return object_wid
    return -1


def object_height(object_name: str) -> float:
    for _ in range(4):
        scene_description = get_latest_detection()
        if scene_description:
            for item in scene_description:
                if object_name.lower() in item["name"].lower():
                    object_height = (item["ymax"] - item["ymin"]) / 1200
                    return object_height
    return -1


async def delay(seconds: int) -> None:
    await asyncio.sleep(seconds)


async def move_in_circle(cw: bool) -> None:
    heading_deg = await get_heading()
    for degrees in range(0, 360, 36):
        await turn_cw(degrees)
        await asyncio.sleep(1)


async def move_forward(distance: float) -> None:
    if await check_connection_with_retry(): await move_in_direction(distance, "forward")


async def move_backward(distance: float) -> None:
    if await check_connection_with_retry(): await move_in_direction(distance, "backward")


async def move_left(distance: float) -> None:
    if await check_connection_with_retry(): await move_in_direction(distance, "left")


async def move_right(distance: float) -> None:
    if await check_connection_with_retry():
        await move_in_direction(distance, "right")


# Move the drone up by increasing altitude (using goto_location)
async def move_up(distance: float):
    if await check_connection_with_retry():
        async for position in drone.telemetry.position():
            current_lat = position.latitude_deg
            current_lon = position.longitude_deg
            current_altitude = position.relative_altitude_m
            absoulte_alt = position.absolute_altitude_m
            break
        #print(f'current_altitude: {current_altitude}\nabsoulte_alt: {absoulte_alt}')
        yaw = await get_heading()
        new_altitude = current_altitude + distance
        new_abs_alt = absoulte_alt + distance
        print(f"Moving up to {new_altitude} meters, from {current_altitude} meters")

        """# Move to the new altitude while keeping the same latitude, longitude, and heading
        #await drone.action.goto_location(current_lat, current_lon, new_altitude, yaw)  # Assuming 0 degrees yaw
        #await asyncio.sleep(25)
        changing_elevation = 1.5 + distance
        await drone.action.set_takeoff_altitude(changing_elevation)
        starting_altitude = 0
        await drone.action.set_return_to_launch_altitude(relative_altitude_m=new_altitude)

        # Use a timeout to wait until the drone reaches the target altitude"""
        #print(f"Moving up by {new_altitude} meters...")
        await drone.action.goto_location(current_lat, current_lon, new_abs_alt, yaw)
        #print("*" * 5)
        await timeout_check(2000, lambda: asyncio.ensure_future(reached_target_altitude(new_altitude, current_altitude)))


# Move the drone down by decreasing altitude (using goto_location)
async def move_down(distance: float):
    if await check_connection_with_retry():
        async for position in drone.telemetry.position():
            current_lat = position.latitude_deg
            current_lon = position.longitude_deg
            current_altitude = position.relative_altitude_m
            absoulte_alt = position.absolute_altitude_m
            break
        """yaw = await get_heading()
        new_altitude = current_altitude - distance
        print(f"Moving down to {new_altitude} meters")

        # Move to the new altitude while keeping the same latitude, longitude, and heading
        await drone.action.goto_location(current_lat, current_lon, new_altitude, yaw)  # Assuming 0 degrees yaw
        await asyncio.sleep(25)"""
        yaw = await get_heading()
        new_altitude = current_altitude - distance
        new_abs_alt = absoulte_alt - distance
        print(f"Moving down to {new_altitude} meters, from {current_altitude} meters")

        """# Move to the new altitude while keeping the same latitude, longitude, and heading
        #await drone.action.goto_location(current_lat, current_lon, new_altitude, yaw)  # Assuming 0 degrees yaw
        #await asyncio.sleep(25)
        changing_elevation = 1.5 + distance
        await drone.action.set_takeoff_altitude(changing_elevation)
        starting_altitude = 0
        await drone.action.set_return_to_launch_altitude(relative_altitude_m=new_altitude)

        # Use a timeout to wait until the drone reaches the target altitude"""
        # print(f"Moving up by {new_altitude} meters...")
        await drone.action.goto_location(current_lat, current_lon, new_abs_alt, yaw)
        # print("*" * 5)
        await timeout_check(2000,
                            lambda: asyncio.ensure_future(reached_target_altitude(new_altitude, current_altitude, "down")))

async def turn_cw(degrees: float) -> None:
    if await check_connection_with_retry():
        heading_deg = await get_heading()
        while degrees >= 360:
            degrees = degrees - 360
        await set_heading_with_velocity(heading_deg + degrees)

async def turn_ccw(degrees: float) -> None:
    if await check_connection_with_retry():
        heading_deg = await get_heading()
        while degrees >= 360:
            degrees = degrees - 360
        await set_heading_with_velocity(heading_deg - degrees)


async def set_heading_with_velocity(yaw_deg):
    global drone

    # Set the offboard mode
    print("Setting offboard mode...")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, yaw_deg))

    try:
        await drone.offboard.start()
        print(f"Rotating to {yaw_deg} degrees heading...")
        await asyncio.sleep(25)
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Offboard mode failed with error: {error._result.result}")

"""async def close_drone():
    print("Closing drone connection...")
    await drone.close()"""

import grpc
from mavsdk import System

async def check_connection():
    global drone
    try:
        # Try to monitor the connection state
        async for state in drone.core.connection_state():
            if state.is_connected:
                print("Drone is connected")
                break
            else:
                print("Drone is disconnected. Reconnecting...")
                await reconnect_drone()
    except grpc._channel._MultiThreadedRendezvous as e:
        print(f"gRPC error: {e}. Attempting to reconnect...")
        await reconnect_drone()

async def reconnect_drone():
    global drone
    try:
        await drone.connect(system_address="udp://:14540")
        print("Reconnected to the drone.")
    except Exception as e:
        print(f"Failed to reconnect: {e}")

async def check_connection_with_retry(retry_limit=12):
    global drone
    retries = 0
    while retries < retry_limit:
        try:
            async for state in drone.core.connection_state():
                if state.is_connected:
                    print("Drone is connected")
                    return True
                else:
                    print("Drone is disconnected. Reconnecting...")
                    await reconnect_drone()
                    retries += 1
                    if retries >= retry_limit:
                        print("Max retry limit reached. Connection failed.")
                        return False
        except grpc._channel._MultiThreadedRendezvous as e:
            if retries >= retry_limit - 2:
                print(f"gRPC error: {e}. Attempting to reconnect...")
            retries += 1
            await reconnect_drone()


def get_latest_ulog():
    log_dir = os.path.expanduser('~/Documents/QGroundControl/Logs')
    list_of_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.ulg')]
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def copy_latest_ulog(destination):
    latest_ulog = get_latest_ulog()
    if latest_ulog:
        shutil.copy(latest_ulog, destination)
        print(f"Copied {latest_ulog} to {destination}")


async def disconnect_drone():
    global drone
    print("Disconnecting drone...")
    await drone.close()  # Close the drone connection gracefully
    print("Drone disconnected")

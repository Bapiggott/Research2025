import time
from typing import List, Optional
import re
import requests
import subprocess
import os
import glob
import ollama
import argparse


###############################################################################
#                             Timer Helper                                    #
###############################################################################
def timer_step(step_name: str, last_time: float) -> float:
    """
    Prints the elapsed time since 'last_time' and returns the current time.
    """
    current_time = time.time()
    print(f"[TIMER] {step_name} took: {current_time - last_time:.6f} seconds")
    return current_time

###############################################################################
#                          Argument Parsing                                   #
###############################################################################
start_time = time.time()  # Initial checkpoint
print("[INFO] Starting the program...")
# Set up argument parser
parser = argparse.ArgumentParser(description="Configure model variables via command-line arguments.")

# Add arguments
parser.add_argument("--llm_model_name", type=str, default="deepseek-r1:8b", help="LLM model name")
parser.add_argument("--api_url", type=str, default="http://localhost:5000", help="API URL for LLM")
parser.add_argument("--vlm_api_url", type=str, default="http://localhost:8889", help="VLM API URL")
parser.add_argument("--vlm_model_name", type=str, default="llama3.2-vision", help="VLM model name")
parser.add_argument("--llm_api_url", type=str, default="http://localhost:8888", help="LLM API URL")

# Parse arguments
args = parser.parse_args()

# Assign values to variables
LLM_MODEL_NAME = args.llm_model_name
api_url = args.api_url
VLM_API_URL = args.vlm_api_url
VLM_MODEL_NAME = args.vlm_model_name
OLLAMA_API_URL = args.llm_api_url

# Output the variables for confirmation
print(f"LLM_MODEL_NAME: {LLM_MODEL_NAME}")
print(f"api_url: {api_url}")
print(f"VLM_API_URL: {VLM_API_URL}")
print(f"VLM_MODEL_NAME: {VLM_MODEL_NAME}")
#curl -X POST http://0.0.0.0:8889/api/pull -d '{"model": "llama3.2-vision", "name": "llama3.2-vision"}'

time_checkpoint = timer_step("Parsing arguments", start_time)

###############################################################################
#                              Class Definitions                              #
###############################################################################
class VLMWrapper:
    def __init__(self, temperature, model_name=VLM_MODEL_NAME):
        self.temperature = temperature
        self.model_name = model_name
        self.temperature = temperature



    def send_request(self, image: str = None, prompt: str = None, stream=False):
        """
        Send a request to the VLM API with an image and a prompt.
        """
        if prompt == None:
            prompt = "Describe briefly what is going on in this camera image taken from a drone"
        try:
            uploads_directory = "/home/liu/bap/uploads"
            # Get the newest file in the directory
            # image = "/home/liu/bap/image2.jpg" #
            image   = get_newest_file(uploads_directory)
            print(image)
            if not image:
                print("No image file found to send to the VLM.")
                return "No image file found."

                print(f"Using image file: {image}")
                # Encode the image as a Base64 string
                encoded_image = encode_image_to_base64(image)

                if not encoded_image:
                    print("Failed to encode image.")
                    return "Image encoding failed."
            else:
                print(f"Using image file: {image}")
                # Encode the image as a Base64 string
                encoded_image = encode_image_to_base64(image)
                if not encoded_image:
                    print("Failed to encode image.")
                    return "Image encoding failed."
            print(f"Sending request to VLM model '{self.model_name}' with image and prompt:\n{prompt}\n")
            #print(f"Encoded Image: {encoded_image}")
            # Make the POST request to the VLM API
            print("---")
            print(self.model_name)
            """response = requests.post(
                f"{OLLAMA_API_URL}/api/generate -d ", #analyze",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "image": [encoded_image],  # Image file path or encoded data "temperature": self.temperature,
                    "temperature": self.temperature,
                    "stream": stream
                },
                timeout=999
            )
            print(f"response: {response}")"""
            from ollama import Client

            #from ollama import Client

            client = Client(host='http://0.0.0.0:8889')


            response = client.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': 'What is in this image?',
                    'images': [image]
                }]
            )

            """response = ollama.chat(
                model=VLM_MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': 'What is in this image?',
                    'images': [image]
                }]
            )"""

            print(f"VLM Response {response}")

            # Check if the response status is OK
            if response.status_code != 200:
                print(f"Error: VLM API responded with status code {response.status_code}")
                print(f"Response content: {response.text}")
                return response

            # Parse the JSON response
            data = response.json()
            response_text = data.get('response', 'No response found')

            # Print the response in the terminal
            print(f"Response from VLM model:\n{response_text}")
            return response_text

        except Exception as e:
            print(f"Request to VLM failed: {e}")



class LLMWrapper:
    def __init__(self, temperature, model_name=LLM_MODEL_NAME):
        self.temperature = temperature
        self.model_name = model_name

    def send_request(self, prompt: str, stream=False):
        """
        Send a request to the API and print the response in the terminal.
        """
        try:
            print(f"Sending request to model '{self.model_name}' with prompt:\n{prompt}\n")

            # Make the POST request to the API
            response = requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": stream,
                },
                timeout=999
            )

            # Check if the response status is OK
            if response.status_code != 200:
                print(f"Error: API responded with status code {response.status_code}")
                print(f"Response content: {response.text}")
                return response

            # Parse the JSON response
            data = response.json()
            text = data.get('response', 'No response found')

            # Print the response in the terminal
            print(f"Response from model:\n{text}")
            return text

        except Exception as e:
            print(f"Request failed: {e}")



class MiniSpecProgram:
    def __init__(self, env: Optional[dict] = None) -> None:
        self.statements: List[str] = []  # To store MiniSpec statements
        self.python_code: List[str] = []  # To store equivalent Python code
        self.env = env if env is not None else {}
        self.indentation_level = 1  # Start with indentation level 1 for 'async def main()'

        # Define abbreviations and their corresponding function names
        self.abbreviations = {
            's': 'scan',
            'sa': 'scan_abstract',
            'o': 'orienting',
            'a': 'approach',
            'g': 'goto',
            'mf': 'move_forward',
            'mb': 'move_backward',
            'ml': 'move_left',
            'mr': 'move_right',
            'mu': 'move_up',
            'md': 'move_down',
            'tc': 'turn_cw',
            'tu': 'turn_ccw',
            'mi': 'move_in_circle',
            'd': 'delay',
            'iv': 'is_visible',
            'ox': 'object_x',
            'oy': 'object_y',
            'ow': 'object_width',
            'oh': 'object_height',
            'od': 'object_dis',
            'p': 'probe',
            'l': 'log',
            'tp': 'take_picture',
            'rp': 're_plan',
        }

        # Define function signatures
        self.functions = {
            'scan': 'scan(object_name: str)',
            'scan_abstract': 'scan_abstract(question: str)',
            'orienting': 'orienting(object_name: str)',
            'approach': 'approach()',
            'goto': 'goto(object_name: str)',
            'move_forward': 'move_forward(distance: int)',
            'move_backward': 'move_backward(distance: int)',
            'move_left': 'move_left(distance: int)',
            'move_right': 'move_right(distance: int)',
            'move_up': 'move_up(distance: int)',
            'move_down': 'move_down(distance: int)',
            'turn_cw': 'turn_cw(degrees: int)',
            'turn_ccw': 'turn_ccw(degrees: int)',
            'move_in_circle': 'move_in_circle(cw: bool)',
            'delay': 'delay(seconds: int)',
            'is_visible': 'is_visible(object_name: str)',
            'object_x': 'object_x(object_name: str)',
            'object_y': 'object_y(object_name: str)',
            'object_width': 'object_width(object_name: str)',
            'object_height': 'object_height(object_name: str)',
            'object_dis': 'object_dis(object_name: str)',
            'probe': 'probe(question: str)',
            'log': 'log(text: str)',
            'take_picture': 'take_picture()',
            're_plan': 're_plan()'
        }

        self.await_function = [
            'connect_drone',
            'ensure_armed_and_taken_off',
            'moving_drone',
            'get_heading',
            'move_forward',
            'move_backward',
            'set_heading_with_velocity',
            'move_left',
            'move_right',
            'changing_elevation',
            'move_up',
            'move_down',
            'turn_cw',
            'turn_ccw',
            'move_in_circle',
            'delay',
            'orienting',
            'approach',
            'scan',
            'scan_abstract',
            'goto'
        ]

    def add_statement(self, statement: str) -> None:
        """Add MiniSpec statement and its Python equivalent."""
        python_stmt = self.translate_to_python(statement)
        if python_stmt:
            self.python_code.append(python_stmt)

    def translate_to_python(self, minispec_stmt: str) -> str:
        import re
        minispec_stmt = minispec_stmt.strip()
        if not minispec_stmt:
            return ""

        # Handle loop syntax
        if minispec_stmt[0].isdigit():
            loop_count = int(minispec_stmt.strip('{').strip())
            return self.get_indent() + f"for _ in range({loop_count}):"

        # Handle conditions
        if minispec_stmt.startswith('?'):
            condition = minispec_stmt[1:].strip()
            condition = condition.replace('&', ' and ').replace('|', ' or ')
            # Process abbreviations in the condition
            condition = self.replace_abbreviations(condition)
            # Find all function calls in the condition
            func_calls = re.findall(r'(\w+\(.*?\))', condition)
            for call in func_calls:
                func_name = call.split('(')[0]
                if func_name in self.await_function:
                    # Replace function call with 'await' prefixed
                    condition = condition.replace(call, f'await {call}')
            return self.get_indent() + f"if {condition}:"

        # Handle return statement
        elif minispec_stmt.startswith('->'):
            return self.get_indent() + f"return {minispec_stmt[2:].strip()}"

        # Handle assignment statements
        if '=' in minispec_stmt:
            lhs, rhs = minispec_stmt.split('=', 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            # Process RHS for function abbreviations
            rhs = self.replace_abbreviations(rhs)
            # Check if RHS is a function call
            func_match = re.match(r'^(\w+)\s*\(.*\)$', rhs)
            if func_match:
                func_name = func_match.group(1)
                if func_name in self.await_function:
                    rhs = 'await ' + rhs
            return self.get_indent() + f"{lhs} = {rhs}"

        # Check for function call with abbreviation
        for abbr, func_name in self.abbreviations.items():
            if minispec_stmt.startswith(abbr + '('):
                # Replace abbreviation with function name
                if func_name in self.await_function:
                    return self.get_indent() + "await " + func_name + '(' + minispec_stmt[len(abbr) + 1:-1] + ')'
                else:
                    return self.get_indent() + func_name + '(' + minispec_stmt[len(abbr) + 1:-1] + ')'

        # Check for function call without arguments
        if minispec_stmt in self.abbreviations:
            func_name = self.abbreviations[minispec_stmt]
            if func_name in self.await_function:
                return self.get_indent() + "await " + func_name + "()"
            else:
                return self.get_indent() + func_name + "()"

        # Process abbreviations in the entire statement
        minispec_stmt = self.replace_abbreviations(minispec_stmt)

        # Handle other statements
        return self.get_indent() + minispec_stmt

    def replace_abbreviations(self, s: str) -> str:
        import re
        for abbr, func_name in self.abbreviations.items():
            # Match abbreviation when it's a standalone word or followed by '('
            pattern = r'\b' + re.escape(abbr) + r'(\b|\()'
            # If func_name is a function without arguments
            if func_name in self.functions and self.functions[func_name].endswith('()'):
                # If followed by '(', keep it (e.g., function call with arguments)
                if re.search(r'\b' + re.escape(abbr) + r'\(', s):
                    replacement = func_name + '('
                else:
                    replacement = func_name + '()'
            else:
                replacement = func_name + r'\1'  # Keep the '(' or word boundary
            s = re.sub(pattern, replacement, s)
        return s

    def get_indent(self) -> str:
        """Get the current indentation level for Python code."""
        return '    ' * self.indentation_level  # 4 spaces per indentation level


    def write_output(self, filename: str) -> None:
        """Write MiniSpec and equivalent Python code to the output file."""
        with open(filename, 'w') as f:
            f.write('import time\n')
            #f.write('from jinja2.nodes import Continue\n\n')
            f.write('from functions_gps import *\n')
            f.write('import asyncio\n')
            f.write('import datetime\n')
            f.write('import json\n')
            f.write('import os\n\n')
            # Include the original MiniSpec code
            escaped_original_code = self.original_code.replace('"""', '\\"\\"\\"')
            f.write(f'original_code = """{escaped_original_code}"""\n')
            f.write('START_SAVING_URL = "http://localhost:5000/start_saving"\n')
            f.write('STOP_SAVING_URL = "http://localhost:5000/stop_saving"\n\n')


            f.write('async def main():\n')
            f.write('    mission_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\n')
            f.write(f'    mission_description = "{filename}"\n')
            f.write('    if mission_description == "":\n')
            f.write('        n = ""\n')
            f.write('    else:\n')
            f.write('        n = \'_\'\n')
            f.write('    mission_directory = f"mission_{mission_description}{n}{mission_timestamp}"\n')
            f.write('    with open(__file__, \'r\') as f_in:\n')
            f.write('        translated_code = f_in.read()\n')
            f.write('    start_data = {\n')
            f.write('        "mission_directory": mission_directory,\n')
            f.write('        "original_code": original_code,\n')
            f.write('        "translated_code": translated_code\n')
            f.write('    }\n\n')
            f.write('    # Notify the server to start saving images to the mission directory\n')
            f.write('    requests.post(START_SAVING_URL, json=start_data)\n\n')



            f.write('    await connect_drone()\n')
            f.write('    await ensure_armed_and_taken_off()\n')
            f.write('    # await return_to_start_position()\n')  # Add this line
            f.write('    # Start the print_status_text task\n')
            f.write('    # status_task = asyncio.create_task(print_status_text(drone))\n\n')
            f.write('    try:\n')
            for line in self.python_code:
                f.write(f"    {line}\n")
            f.write('\n')
            """f.write('    finally:\n')
            f.write('        await stop_offboard_mode()\n')
            f.write('        status_task.cancel()\n')
            f.write('        try:\n')
            f.write('            await status_task\n')
            f.write('        except asyncio.CancelledError:\n')
            f.write('            pass\n\n')"""
            f.write('    except:\n')
            f.write('        print("")\n')
            f.write('    await land_drone()\n')
            f.write('    requests.post(STOP_SAVING_URL)\n')
            f.write('    print("STOP_SAVING_URL")\n\n')


            # Include code to run main() and log the output
            f.write('if __name__ == \'__main__\':\n')
            f.write('    result = asyncio.run(main())\n')
            f.write('    current_datetime = datetime.datetime.now().isoformat()\n')
            f.write('    with open(__file__, \'r\') as f_in:\n')
            f.write('        translated_code = f_in.read()\n')
            f.write('    log_data = {\n')
            f.write('        \'date\': current_datetime,\n')
            f.write('        \'original_code\': original_code,\n')
            f.write('        \'translated_code\': translated_code,\n')
            f.write('        \'output\': result\n')
            f.write('    }\n')
            f.write('    with open(\'execution_log.json\', \'a\') as log_file:\n')
            f.write('        log_file.write(json.dumps(log_data) + \'\\n\')\n')
            f.write('    destination_folder = "./saved_logs"\n')
            f.write('    print("end")\n')
            f.write('    time.sleep(15)\n')
            f.write('    os._exit(0)\n')

    def parse(self, code: str) -> None:
        """Simulate parsing and add statements."""
        self.original_code = code
        statement = ""  # To accumulate characters into a statement
        i = 0
        while i < len(code):
            char = code[i]
            if char == ';':  # End of a statement
                if statement.strip():
                    self.add_statement(statement.strip())
                    statement = ""  # Reset statement buffer
                i += 1
            elif char == '{':
                if statement.strip():
                    self.add_statement(statement.strip())
                    statement = ""  # Reset statement buffer
                self.indentation_level += 1
                i += 1
            elif char == '}':
                if statement.strip():
                    self.add_statement(statement.strip())
                    statement = ""  # Reset statement buffer
                self.indentation_level -= 1
                i += 1
            else:
                statement += char
                i += 1
        if statement.strip():
            self.add_statement(statement.strip())  # Add final statement


def get_latest_detection(api_url):
    """
    Fetches and prints only the latest detection's 'detections' array from the given API URL.
    """
    try:
        response = requests.get(f"{api_url}/detections/latest")

        # Check if the request was successful
        if response.status_code == 200:
            latest_detection = response.json()
            detections = latest_detection.get('detections', [])
            return detections
        elif response.status_code == 404:
            return []
        else:
            print(f"Error: Received unexpected status code {response.status_code}")
            print("Response:", response.text)
            return "error"
    except Exception as e:
        print(f"Failed to fetch the latest detection: {e}")



def log_to_file(prompt: str, response: str, log_file: str = "missions_log.txt"):
    """
    Append the prompt and response to a log file.

    Args:
        prompt (str): The user's input prompt.
        response (str): The model's response.
        log_file (str): The file where logs are stored.
    """
    with open(log_file, "a") as file:
        file.write("Mission Log:\n")
        file.write(f"Prompt: {prompt}\n")
        file.write(f"Response: {response}\n")
        file.write(f"Model Name: {LLM_MODEL_NAME}\n")

        file.write("-" * 40 + "\n")
def get_newest_file(directory: str) -> str:
    """
    Get the newest file from the specified directory.

    Args:
        directory (str): The path to the directory.

    Returns:
        str: The path to the newest file, or None if no files are found.
    """
    # Get all files in the directory
    files = glob.glob(os.path.join(directory, '*'))  # You can adjust the pattern if needed
    if not files:
        print("No files found in the directory.")
        return None

    # Find the newest file by modification time
    newest_file = max(files, key=os.path.getmtime)
    return newest_file

def fetch_minispec_from_llm1(short_prompt: str, prompt: str):
    """Fetch MiniSpec plan dynamically from the LLM."""
    print("Fetching MiniSpec plan from LLM...")
    try:
        llm_response = llm.send_request(prompt).strip()  # Adjust this based on LLM API's actual return format
        print("-------")
        vlm_reponse = vlm.send_request()
        print("*********")
        print(f"LLM's response: {llm_response}")
        print(f"LLM's response stripped: {llm_response}")
        print(f"VLM's response: {vlm_reponse}")
        log_to_file(short_prompt, llm_response)
        print("########")

        # Assuming response is returned directly as text for simplicity
        return llm_response.strip(), vlm_reponse
    except Exception as e:
        print(f"Error fetching MiniSpec plan: {e}")
        return ""

def fetch_minispec_from_llm(llm, vlm, short_prompt: str, prompt: str):
    """Fetch MiniSpec plan dynamically from the LLM."""
    print("Fetching MiniSpec plan from LLM...")
    try:
        # We can directly use the llm and vlm objects that were passed in
        print("Sending request to LLM...")
        llm_response = llm.send_request(prompt)
        llm_response_uncleaned = llm_response
        print(f"LLM's response: {llm_response}")
        llm_response = re.sub(
            r"<think>.*?</think>\n?",
            "",
            llm_response,
            flags=re.DOTALL  # DOTALL so '.' matches newlines as well
        ).strip()
        llm_response = re.sub(
            r"(?m)^response:.*",  # Match 'response:' at the start of a line
            "",  # Replace with nothing
            llm_response
        )
        print("Sending request to VLM...")
        vlm_response = vlm.send_request()
        print(f"LLM's response: {llm_response}")
        print(f"VLM's response: {vlm_response}")

        # Log the short_prompt + llm_response to a file
        log_to_file(short_prompt, llm_response)

        # Return both the LLM's text + VLM's text
        return llm_response, vlm_response
    except Exception as e:
        print(f"Error fetching MiniSpec plan: {e}")
        return "", ""

def log_to_file(prompt: str, response: str, log_file: str = "missions_log.txt"):
    """
    Append the prompt and response to a log file.

    Args:
        prompt (str): The user's input prompt.
        response (str): The model's response.
        log_file (str): The file where logs are stored.
    """
    with open(log_file, "a") as file:
        file.write("Mission Log:\n")
        file.write(f"Prompt: {prompt}\n")
        file.write(f"Response: {response}\n")
        file.write("-" * 40 + "\n")

import base64

def encode_image_to_base64(image_path):
    """
    Encode an image to a Base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_string
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

class SecondCodeHandler:
    """
    This class wraps all the logic that was originally in the
    `if __name__ == '__main__':` block.
    """
    def __init__(self):
        # Create your LLM and VLM wrappers (with any default config you need)
        self.llm = LLMWrapper(temperature=0.7)
        self.vlm = VLMWrapper(temperature=0.7)
        # If you need any other setup, do it here

    def run(self, prompt: str) -> str:
        """
        Orchestrate the entire process:
          1. Build a 'scene_desc' or fetch it from the YOLO server
          2. Build a 'big_prompt' to pass to the LLM
          3. Call `fetch_minispec_from_llm` to get the MiniSpec code
          4. Parse the code into a Python script
          5. Write that Python script to disk (output_llm.py)
          6. Return or log the result
        """
        step_start1 = time.time()
        # Just an example scene_desc (replace with real data if needed)
        scene_desc = get_latest_detection(api_url)

        # Build a longer prompt that includes instructions
        big_prompt = (
            "You are a robot pilot and you should follow the user's instructions to generate a MiniSpec plan "
            "to fulfill the task or give advice on user's input if it's not clear or not reasonable.\n\n"
            "Your response should carefully consider the 'system skills description', the 'scene description', "
            "the 'task description', and both the 'previous response' and the 'execution status' if they are provided.\n"
            "The 'system skills description' describes the system's capabilities which include low-level and high-level skills. "
            "Low-level skills, while fixed, offer direct function calls to control the robot and acquire vision information. "
            "High-level skills, built with our language 'MiniSpec', are more flexible and can be used to build more complex skills. "
            "Whenever possible, please prioritize the use of high-level skills, invoke skills using their designated abbreviations, "
            "and ensure that 'object_name' refers to a specific type of object. If a skill has no argument, you can call it without parentheses.\n\n"
            "Description of the two skill sets:\n"
            "- High-level skills:\n"
            "abbr:s,name:scan,definition:8{?iv($1)==True{->True}tc(45)}->False;,args:[object_name:str],description:Rotate to find object $1 when it's *not* in current scene\n"
            "abbr:sa,name:scan_abstract,definition:8{_1=p($1);?_1!=False{->_1}tc(45)}->False;,args:[question:str],description:Rotate to find an abstract object by a description $1 when it's *not* in current scene\n"
            "abbr:o,name:orienting,definition:4{_1=ox($1);?_1>0.6{tc(15)};?_1<0.4{tu(15)};_2=ox($1);?_2<0.6&_2>0.4{->True}}->False;,args:[object_name:str],description:Rotate to align with object $1\n"
            "abbr:a,name:approach,definition:mf(100);,args:[],description:Approach forward\n"
            "abbr:g,name:goto,definition:orienting($1);approach();,args:[object_name:str],description:Go to object $1\n\n"
            "- Low-level skills:\n"
            "abbr:mf,name:move_forward,args:[distance:int],description:Move forward by a distance\n"
            "abbr:mb,name:move_backward,args:[distance:int],description:Move backward by a distance\n"
            "abbr:ml,name:move_left,args:[distance:int],description:Move left by a distance\n"
            "abbr:mr,name:move_right,args:[distance:int],description:Move right by a distance\n"
            "abbr:mu,name:move_up,args:[distance:int],description:Move up by a distance\n"
            "abbr:md,name:move_down,args:[distance:int],description:Move down by a distance\n"
            "abbr:tc,name:turn_cw,args:[degrees:int],description:Rotate clockwise/right by certain degrees\n"
            "abbr:tu,name:turn_ccw,args:[degrees:int],description:Rotate counterclockwise/left by certain degrees\n"
            "abbr:mi,name:move_in_circle,args:[cw:bool],description:Move in circle in cw/ccw\n"
            "abbr:d,name:delay,args:[milliseconds:int],description:Wait for specified microseconds\n"
            "abbr:iv,name:is_visible,args:[object_name:str],description:Check the visibility of target object\n"
            "abbr:ox,name:object_x,args:[object_name:str],description:Get object's X-coordinate in (0,1)\n"
            "abbr:oy,name:object_y,args:[object_name:str],description:Get object's Y-coordinate in (0,1)\n"
            "abbr:ow,name:object_width,args:[object_name:str],description:Get object's width in (0,1)\n"
            "abbr:oh,name:object_height,args:[object_name:str],description:Get object's height in (0,1)\n"
            "abbr:od,name:object_dis,args:[object_name:str],description:Get object's distance in cm\n"
            "abbr:p,name:probe,args:[question:str],description:Probe the LLM for reasoning\n"
            "abbr:l,name:log,args:[text:str],description:Output text to console\n"
            "abbr:tp,name:take_picture,args:[],description:Take a picture\n"
            "abbr:rp,name:re_plan,args:[],description:Replanning\n\n"
            "The 'scene description' is an object list of the current view, containing their names with ID, location, and size (location and size are floats between 0~1). "
            "This may not be useful if the task is about the environment outside the view.\n"
            "The 'task description' is a natural language sentence, describing the user's instructions. It may start with '[A]' or '[Q]'. "
            "'[A]' sentences mean you should generate an execution plan for the robot. '[Q]' sentences mean you should use 'log' to show a literal answer at the end of the plan execution. "
            "Please carefully reason about the 'task description', you should interpret it and generate a detailed multi-step plan to achieve it as much as you can\n"
            "The 'execution history' is the actions have been taken from previous response. When they are provided, that means the robot is doing replanning, "
            "and the user wants to continue the task based on the task and history. You should reason about the 'execution history' and generate a new response accordingly.\n\n"
            "Here is a list of example 'response' for different 'scene description' and 'task description', and their explanations:\n"
            "Example 1:\n"
            "Scene: []\n"
            "Task: [A] Find a bottle, tell me it's height and take a picture of it.\n"
            "Reason: no bottle instance in the scene, so we use scan to find bottle, then go and use object_height to get the height and log to output the height, finally use picture to take a picture of the bottle\n"
            "Response: ?s('bottle')==True{g('bottle');_2=oh('bottle');l(_2);tp};\n\n"
            "Example 2:\n"
            "Scene: [apple_5]\n"
            "Task: [A] Find an apple.\n"
            "Reason: there is an apple instance in the scene, we just go to the apple_5\n"
            "Response: g('apple_5');\n\n"
            "Example 3:\n"
            "Scene: [apple_3]\n"
            "Task: [Q] Is there an apple and an orange on your left?\n"
            "Reason: turn left 90 degrees, then use is_visible to check whether there is an apple on your left\n"
            "Response: tu(90);?iv('apple')==True&iv('orange'){l('Yes');->True}l('No');->False;\n\n"
            "Example 4:\n"
            "Scene: [chair_13,laptop_2]\n"
            "Task: [A] Go to the chair behind you.\n"
            "Reason: the chair_13 is not the target because we want the one behind you. So we turn 180 degrees then go to the general object chair, since chair is a large object, we use 80cm as the distance.\n"
            "Response: tc(180);g('chair');\n\n"
            "Example 5:\n"
            "Scene: [chair_3,laptop_1,bottle_5]\n"
            "Task: [A] Find and go any edible object.\n"
            "Reason: edible object is abstract and there is no edible object in the scene, so we use scan_abstract to find the edible object\n"
            "Response: _1=sa('Any edible target here?');?_1!=False{g(_1,10)};\n\n"
            "Example 6:\n"
            "Scene: [chair_3,laptop_9]\n"
            "Task: [A] Turn around with 30 degrees step until you can see some animal.\n"
            "Reason: we use loop and probe to find animal\n"
            "Response: 12{_1=p('Any animal target here?');?_1!=False{l(_1);->True}tc(30)}->False;\n\n"
            "Example 7:\n"
            "Scene: [chair_3,laptop_9]\n"
            "Task: [A] If you can see a chair, go find a person, else go find an orange.\n"
            "Reason: From the scene, we can see a chair, so we use scan to find a person. Since person is a large object, we use 60cm as the distance\n"
            "Response: _1=s('person');?_1==True{g('person');->True}->False;\n\n"
            "Example 8:\n"
            "Scene: [chair_3,laptop_9]\n"
            "Task: [A] Go to \n"
            "Reason: The task is too vague, so we use log to output the advice\n"
            "Response: l('Please give me more information.');\n\n"
            "Example 9:\n"
            "Scene: [chair_1 x:0.58 y:0.5 width:0.43 height:0.7, apple_1 x:0.6 y:0.3 width:0.08 height:0.09]\n"
            "Task: [A] Turn around and go to the apple\n"
            "Reason: after turning around, we will do replan. We found that the chair is blocking the apple, so we use moving_up to get over the chair and then go to the apple\n"
            "Response: mu(40);g('apple');\n\n"
            "Example 10:\n"
            "Scene: [apple_1 x:0.34 y:0.3 width:0.09 height:0.1, apple_2 x:0.3 y:0.5 width:0.15 height:0.12]\n"
            "Task: [A] Go to biggest apple\n"
            "Reason: from the scene, we tell directly that apple_2 is the biggest apple\n"
            "Response: g('apple_2');\n\n"
            "Here is the 'scene description':\n"
            f"{scene_desc}\n\n"
            "Here is the 'task description':\n"
            f"[A] {prompt}\n\n"
            "Here is the 'execution history' (has value if replanning):\n"
            "None\n\n"
            "Please generate the response only with a single sentence of MiniSpec program.\n"
            "'response':"
        )



        print(f"Prompt: {prompt}")
        step_start1 = timer_step("Building big_prompt", step_start1)



        # 1) Call the LLM + VLM to get the MiniSpec code
        minispec_code, vlm_answer = fetch_minispec_from_llm(
            self.llm,
            self.vlm,
            short_prompt=prompt,
            prompt=big_prompt
        )
        step_start1 = timer_step("Fetching MiniSpec from LLM", step_start1)

        # 2) Parse and write out the code
        program = MiniSpecProgram()
        program.parse(minispec_code)
        step_start1 = timer_step("Parsing MiniSpec code", step_start1)
        output_filename = 'output_llm.py'
        program.write_output(output_filename)
        step_start1 = timer_step("Writing Python code to disk", step_start1)
        print(f"Program output written to '{output_filename}'")

        # For a simple return, we can just return the raw MiniSpec code
        # or a short message about success.
        test = True
        if test:
            run_start = time.time()
            try:
                print(f"Executing '{output_filename}'...")
                run_end = time.time()
                result = subprocess.run(['python3', output_filename], check=True)
                print(f"Execution of '{output_filename}' completed successfully.")
                print(f"[TIMER] Running output_llm.py took: {run_end - run_start:.6f} seconds")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while executing '{output_filename}': {e}")
        return f"MiniSpec code:\n{minispec_code}\n\n(VLM Answer: {vlm_answer})"


# Example use case

if __name__ == '__main__':
    # Timer: Start
    step_start = time.time()
    prompt = "Turn 180 degrees then move forward"
    scene_desc = "example" # get_latest_detection(api_url)
    big_prompt = (
        "You are a robot pilot and you should follow the user's instructions to generate a MiniSpec plan "
        "to fulfill the task or give advice on user's input if it's not clear or not reasonable.\n\n"
        "Your response should carefully consider the 'system skills description', the 'scene description', "
        "the 'task description', and both the 'previous response' and the 'execution status' if they are provided.\n"
        "The 'system skills description' describes the system's capabilities which include low-level and high-level skills. "
        "Low-level skills, while fixed, offer direct function calls to control the robot and acquire vision information. "
        "High-level skills, built with our language 'MiniSpec', are more flexible and can be used to build more complex skills. "
        "Whenever possible, please prioritize the use of high-level skills, invoke skills using their designated abbreviations, "
        "and ensure that 'object_name' refers to a specific type of object. If a skill has no argument, you can call it without parentheses.\n\n"
        "Description of the two skill sets:\n"
        "- High-level skills:\n"
        "abbr:s,name:scan,definition:8{?iv($1)==True{->True}tc(45)}->False;,args:[object_name:str],description:Rotate to find object $1 when it's *not* in current scene\n"
        "abbr:sa,name:scan_abstract,definition:8{_1=p($1);?_1!=False{->_1}tc(45)}->False;,args:[question:str],description:Rotate to find an abstract object by a description $1 when it's *not* in current scene\n"
        "abbr:o,name:orienting,definition:4{_1=ox($1);?_1>0.6{tc(15)};?_1<0.4{tu(15)};_2=ox($1);?_2<0.6&_2>0.4{->True}}->False;,args:[object_name:str],description:Rotate to align with object $1\n"
        "abbr:a,name:approach,definition:mf(100);,args:[],description:Approach forward\n"
        "abbr:g,name:goto,definition:orienting($1);approach();,args:[object_name:str],description:Go to object $1\n\n"
        "- Low-level skills:\n"
        "abbr:mf,name:move_forward,args:[distance:int],description:Move forward by a distance\n"
        "abbr:mb,name:move_backward,args:[distance:int],description:Move backward by a distance\n"
        "abbr:ml,name:move_left,args:[distance:int],description:Move left by a distance\n"
        "abbr:mr,name:move_right,args:[distance:int],description:Move right by a distance\n"
        "abbr:mu,name:move_up,args:[distance:int],description:Move up by a distance\n"
        "abbr:md,name:move_down,args:[distance:int],description:Move down by a distance\n"
        "abbr:tc,name:turn_cw,args:[degrees:int],description:Rotate clockwise/right by certain degrees\n"
        "abbr:tu,name:turn_ccw,args:[degrees:int],description:Rotate counterclockwise/left by certain degrees\n"
        "abbr:mi,name:move_in_circle,args:[cw:bool],description:Move in circle in cw/ccw\n"
        "abbr:d,name:delay,args:[milliseconds:int],description:Wait for specified microseconds\n"
        "abbr:iv,name:is_visible,args:[object_name:str],description:Check the visibility of target object\n"
        "abbr:ox,name:object_x,args:[object_name:str],description:Get object's X-coordinate in (0,1)\n"
        "abbr:oy,name:object_y,args:[object_name:str],description:Get object's Y-coordinate in (0,1)\n"
        "abbr:ow,name:object_width,args:[object_name:str],description:Get object's width in (0,1)\n"
        "abbr:oh,name:object_height,args:[object_name:str],description:Get object's height in (0,1)\n"
        "abbr:od,name:object_dis,args:[object_name:str],description:Get object's distance in cm\n"
        "abbr:p,name:probe,args:[question:str],description:Probe the LLM for reasoning\n"
        "abbr:l,name:log,args:[text:str],description:Output text to console\n"
        "abbr:tp,name:take_picture,args:[],description:Take a picture\n"
        "abbr:rp,name:re_plan,args:[],description:Replanning\n\n"
        "The 'scene description' is an object list of the current view, containing their names with ID, location, and size (location and size are floats between 0~1). "
        "This may not be useful if the task is about the environment outside the view.\n"
        "The 'task description' is a natural language sentence, describing the user's instructions. It may start with '[A]' or '[Q]'. "
        "'[A]' sentences mean you should generate an execution plan for the robot. '[Q]' sentences mean you should use 'log' to show a literal answer at the end of the plan execution. "
        "Please carefully reason about the 'task description', you should interpret it and generate a detailed multi-step plan to achieve it as much as you can\n"
        "The 'execution history' is the actions have been taken from previous response. When they are provided, that means the robot is doing replanning, "
        "and the user wants to continue the task based on the task and history. You should reason about the 'execution history' and generate a new response accordingly.\n\n"
        "Here is a list of example 'response' for different 'scene description' and 'task description', and their explanations:\n"
        "Example 1:\n"
        "Scene: []\n"
        "Task: [A] Find a bottle, tell me it's height and take a picture of it.\n"
        "Reason: no bottle instance in the scene, so we use scan to find bottle, then go and use object_height to get the height and log to output the height, finally use picture to take a picture of the bottle\n"
        "Response: ?s('bottle')==True{g('bottle');_2=oh('bottle');l(_2);tp};\n\n"
        "Example 2:\n"
        "Scene: [apple_5]\n"
        "Task: [A] Find an apple.\n"
        "Reason: there is an apple instance in the scene, we just go to the apple_5\n"
        "Response: g('apple_5');\n\n"
        "Example 3:\n"
        "Scene: [apple_3]\n"
        "Task: [Q] Is there an apple and an orange on your left?\n"
        "Reason: turn left 90 degrees, then use is_visible to check whether there is an apple on your left\n"
        "Response: tu(90);?iv('apple')==True&iv('orange'){l('Yes');->True}l('No');->False;\n\n"
        "Example 4:\n"
        "Scene: [chair_13,laptop_2]\n"
        "Task: [A] Go to the chair behind you.\n"
        "Reason: the chair_13 is not the target because we want the one behind you. So we turn 180 degrees then go to the general object chair, since chair is a large object, we use 80cm as the distance.\n"
        "Response: tc(180);g('chair');\n\n"
        "Example 5:\n"
        "Scene: [chair_3,laptop_1,bottle_5]\n"
        "Task: [A] Find and go any edible object.\n"
        "Reason: edible object is abstract and there is no edible object in the scene, so we use scan_abstract to find the edible object\n"
        "Response: _1=sa('Any edible target here?');?_1!=False{g(_1,10)};\n\n"
        "Example 6:\n"
        "Scene: [chair_3,laptop_9]\n"
        "Task: [A] Turn around with 30 degrees step until you can see some animal.\n"
        "Reason: we use loop and probe to find animal\n"
        "Response: 12{_1=p('Any animal target here?');?_1!=False{l(_1);->True}tc(30)}->False;\n\n"
        "Example 7:\n"
        "Scene: [chair_3,laptop_9]\n"
        "Task: [A] If you can see a chair, go find a person, else go find an orange.\n"
        "Reason: From the scene, we can see a chair, so we use scan to find a person. Since person is a large object, we use 60cm as the distance\n"
        "Response: _1=s('person');?_1==True{g('person');->True}->False;\n\n"
        "Example 8:\n"
        "Scene: [chair_3,laptop_9]\n"
        "Task: [A] Go to \n"
        "Reason: The task is too vague, so we use log to output the advice\n"
        "Response: l('Please give me more information.');\n\n"
        "Example 9:\n"
        "Scene: [chair_1 x:0.58 y:0.5 width:0.43 height:0.7, apple_1 x:0.6 y:0.3 width:0.08 height:0.09]\n"
        "Task: [A] Turn around and go to the apple\n"
        "Reason: after turning around, we will do replan. We found that the chair is blocking the apple, so we use moving_up to get over the chair and then go to the apple\n"
        "Response: mu(40);g('apple');\n\n"
        "Example 10:\n"
        "Scene: [apple_1 x:0.34 y:0.3 width:0.09 height:0.1, apple_2 x:0.3 y:0.5 width:0.15 height:0.12]\n"
        "Task: [A] Go to biggest apple\n"
        "Reason: from the scene, we tell directly that apple_2 is the biggest apple\n"
        "Response: g('apple_2');\n\n"
        "Here is the 'scene description':\n"
        f"{scene_desc}\n\n"
        "Here is the 'task description':\n"
        f"[A] {prompt}\n\n"
        "Here is the 'execution history' (has value if replanning):\n"
        "None\n\n"
        "Please generate the response only with a single sentence of MiniSpec program.\n"
        "'response':"
    )
    step_start = timer_step("Building big_prompt", step_start)

    llm = LLMWrapper(temperature=0.7)
    vlm = VLMWrapper(temperature=0.7)
    step_start = timer_step("Creating LLM & VLM wrappers", step_start)

    minispec_code, vlm_answer = fetch_minispec_from_llm(prompt, big_prompt)
    step_start = timer_step("Fetching MiniSpec from LLM & VLM", step_start)

    print(f"MiniSpec code:\n{minispec_code}\n\n(VLM Answer: {vlm_answer})")
    program = MiniSpecProgram()
    print("Parsing MiniSpec code...")
    program.parse(minispec_code)
    step_start = timer_step("Parsing MiniSpec code", step_start)

    output_filename = 'output_llm.py'
    print(f"Writing MiniSpec code to '{output_filename}'...")
    program.write_output(output_filename)
    step_start = timer_step("Writing MiniSpec code", step_start)
    print(f"Program output written to '{output_filename}'")
    test = True
    if test:
        run_start = time.time()
        try:
            print(f"Executing '{output_filename}'...")
            result = subprocess.run(['python3', output_filename], check=True)
            run_end = time.time()
            print(f"Execution of '{output_filename}' completed successfully.")
            print(f"[TIMER] Running output_llm.py took: {run_end - run_start:.6f} seconds")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing '{output_filename}': {e}")

    """for idx, code in enumerate(codes, start=1):
        program = MiniSpecProgram()
        program.parse(code)
        output_filename = f'output__{idx}.py'
        program.write_output(output_filename)
        print(f"Program output written to '{output_filename}'")

        try:
            print(f"Executing '{output_filename}'...")
            result = subprocess.run(['python3', output_filename], check=True)
            print(f"Execution of '{output_filename}' completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing '{output_filename}': {e}")
        print("Waiting...")
        time.sleep(100)

        # Reset the program state for the next code snippet
        program.statements = []
        program.python_code = []
        program.indentation_level = 1"""

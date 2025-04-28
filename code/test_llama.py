import requests

OLLAMA_API_URL = "http://0.0.0.0:8888"  # API URL
MODEL_NAME = 'llama3.1:8b'  # Model name

class LLMWrapper:
    def __init__(self, temperature, model_name=MODEL_NAME):
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
                return

            # Parse the JSON response
            data = response.json()
            text = data.get('response', 'No response found')

            # Print the response in the terminal
            print(f"Response from model:\n{text}")

        except Exception as e:
            print(f"Request failed: {e}")

# Example usage
if __name__ == "__main__":
    llm = LLMWrapper(temperature=0.001)  # You can adjust the temperature as needed
    prompt = """
	Prompt: You are a robot pilot and you should follow the user's instructions to generate a MiniSpec plan to fulfill the task or give advice on user's input if it's not clear or not reasonable.

Your response should carefully consider the 'system skills description', the 'scene description', the 'task description', and both the 'previous response' and the 'execution status' if they are provided.
The 'system skills description' describes the system's capabilities which include low-level and high-level skills. Low-level skills, while fixed, offer direct function calls to control the robot and acquire vision information. High-level skills, built with our language 'MiniSpec', are more flexible and can be used to build more complex skills. Whenever possible, please prioritize the use of high-level skills, invoke skills using their designated abbreviations, and ensure that 'object_name' refers to a specific type of object. If a skill has no argument, you can call it without parentheses.

Description of the two skill sets:
- High-level skills:
abbr:s,name:scan,definition:8{?iv($1)==True{->True}tc(45)}->False;,args:[object_name:str],description:Rotate to find object $1 when it's *not* in current scene
abbr:sa,name:scan_abstract,definition:8{_1=p($1);?_1!=False{->_1}tc(45)}->False;,args:[question:str],description:Rotate to find an abstract object by a description $1 when it's *not* in current scene
abbr:o,name:orienting,definition:4{_1=ox($1);?_1>0.6{tc(15)};?_1<0.4{tu(15)};_2=ox($1);?_2<0.6&_2>0.4{->True}}->False;,args:[object_name:str],description:Rotate to align with object $1
abbr:a,name:approach,definition:mf(100);,args:[],description:Approach forward
abbr:g,name:goto,definition:orienting($1);approach();,args:[object_name:str],description:Go to object $1

- Low-level skills:
abbr:mf,name:move_forward,args:[distance:int],description:Move forward by a distance
abbr:mb,name:move_backward,args:[distance:int],description:Move backward by a distance
abbr:ml,name:move_left,args:[distance:int],description:Move left by a distance
abbr:mr,name:move_right,args:[distance:int],description:Move right by a distance
abbr:mu,name:move_up,args:[distance:int],description:Move up by a distance
abbr:md,name:move_down,args:[distance:int],description:Move down by a distance
abbr:tc,name:turn_cw,args:[degrees:int],description:Rotate clockwise/right by certain degrees
abbr:tu,name:turn_ccw,args:[degrees:int],description:Rotate counterclockwise/left by certain degrees
abbr:mi,name:move_in_circle,args:[cw:bool],description:Move in circle in cw/ccw
abbr:d,name:delay,args:[milliseconds:int],description:Wait for specified microseconds
abbr:iv,name:is_visible,args:[object_name:str],description:Check the visibility of target object
abbr:ox,name:object_x,args:[object_name:str],description:Get object's X-coordinate in (0,1)
abbr:oy,name:object_y,args:[object_name:str],description:Get object's Y-coordinate in (0,1)
abbr:ow,name:object_width,args:[object_name:str],description:Get object's width in (0,1)
abbr:oh,name:object_height,args:[object_name:str],description:Get object's height in (0,1)
abbr:od,name:object_dis,args:[object_name:str],description:Get object's distance in cm
abbr:p,name:probe,args:[question:str],description:Probe the LLM for reasoning
abbr:l,name:log,args:[text:str],description:Output text to console
abbr:tp,name:take_picture,args:[],description:Take a picture
abbr:rp,name:re_plan,args:[],description:Replanning


The 'scene description' is an object list of the current view, containing their names with ID, location, and size (location and size are floats between 0~1). This may not be useful if the task is about the environment outside the view.
The 'task description' is a natural language sentence, describing the user's instructions. It may start with "[A]" or "[Q]". "[A]" sentences mean you should generate an execution plan for the robot. "[Q]" sentences mean you should use 'log' to show a literal answer at the end of the plan execution. Please carefully reason about the 'task description', you should interpret it and generate a detailed multi-step plan to achieve it as much as you can 
The 'execution history' is the actions have been taken from previous response. When they are provided, that means the robot is doing replanning, and the user wants to continue the task based on the task and history. You should reason about the 'execution history' and generate a new response accordingly.

Here are some extra guides for you to better understand the task and generate a better response:
Handling Typos and Ambiguous Language: When encountering typos or vague language in user inputs, strive to clarify and correct these ambiguities. If a typo is evident, adjust the input to reflect the most likely intended meaning. For vague or unclear expressions, seek additional information from the user by returning a question to the user.
Analytical Scene and Task Assessment: Evaluate the scene and task critically. If a specific condition or requirement is clearly deduced from the scene description, generate the corresponding action directly.
Relevance to Current Scene: When considering tasks, assess their relevance to the current scene. If a task does not pertain to the existing scene, disregard any details from the current scene in formulating your response.
Extraction of Key Information: Extract 'object_name' arguments or answers to questions primarily from the 'scene description'. If the necessary information is not present in the current scene, especially when the user asks the drone to move first or check somewhere else, use the probe 'p,' followed by the 'question', to determine the 'object_name' or answer subsequently.
Handling Replan: When previous response and execution status are provided, it means part of the task has been executed. In this case, the system should replan the remaining task based on the current scene and the previous response.

Here is a list of example 'response' for different 'scene description' and 'task description', and their explanations:
Example 1:
Scene: []
Task: [A] Find a bottle, tell me it's height and take a picture of it.
Reason: no bottle instance in the scene, so we use scan to find bottle, then go and use object_height to get the height and log to output the height, finally use picture to take a picture of the bottle
Response: ?s('bottle')==True{g('bottle');_2=oh('bottle');l(_2);tp};

Example 2:
Scene: [apple_5]
Task: [A] Find an apple.
Reason: there is an apple instance in the scene, we just go to the apple_5
Response: g('apple_5');

Example 3:
Scene: [apple_3]
Task: [Q] Is there an apple and an orange on your left?
Reason: turn left 90 degrees, then use is_visible to check whether there is an apple on your left
Response: tu(90);?iv('apple')==True&iv('orange'){l('Yes');->True}l('No');->False;

Example 4:
Scene: [chair_13,laptop_2]
Task: [A] Go to the chair behind you.
Reason: the chair_13 is not the target because we want the one behind you. So we turn 180 degrees then go to the general object chair, since chair is a large object, we use 80cm as the distance.
Response: tc(180);g('chair');

Example 5:
Scene: [chair_3,laptop_1,bottle_5]
Task: [A] Find and go any edible object.
Reason: edible object is abstract and there is no edible object in the scene, so we use scan_abstract to find the edible object
Response: _1=sa('Any edible target here?');?_1!=False{g(_1,10)};

Example 6:
Scene: [chair_3,laptop_9]
Task: [A] Turn around with 30 degrees step until you can see some animal.
Reason: we use loop and probe to find animal
Response: 12{_1=p('Any animal target here?');?_1!=False{l(_1);->True}tc(30)}->False;

Example 7:
Scene: [chair_3,laptop_9]
Task: [A] If you can see a chair, go find a person, else go find an orange.
Reason: From the scene, we can see a chair, so we use scan to find a person. Since person is a large object, we use 60cm as the distance
Response: _1=s('person');?_1==True{g('person');->True}->False;

Example 8:
Scene: [chair_3,laptop_9]
Task: [A] Go to 
Reason: The task is too vague, so we use log to output the advice
Response: l('Please give me more information.');

Example 9:
Scene: [chair_1 x:0.58 y:0.5 width:0.43 height:0.7, apple_1 x:0.6 y:0.3 width:0.08 height:0.09]
Task: [A] Turn around and go to the apple
Reason: after turning around, we will do replan. We found that the chair is blocking the apple, so we use moving_up to get over the chair and then go to the apple
Response: mu(40);g('apple');

Example 10:
Scene: [apple_1 x:0.34 y:0.3 width:0.09 height:0.1, apple_2 x:0.3 y:0.5 width:0.15 height:0.12]
Task: [A] Go to biggest apple
Reason: from the scene, we tell directly that apple_2 is the biggest apple
Response: g('apple_2');

Here is the 'scene description':
[person_1 x:0.21 y:0.50 width:0.42 height:0.87, tv_2 x:0.35 y:0.39 width:0.23 height:0.26, tv_4 x:0.04 y:0.33 width:0.08 height:0.12, chair_3 x:0.49 y:0.82 width:0.67 height:0.31, backpack_5 x:0.89 y:0.82 width:0.23 height:0.37, laptop_6 x:0.05 y:0.34 width:0.09 height:0.14, chair_5 x:0.74 y:0.79 width:0.39 height:0.40, book_7 x:0.54 y:0.62 width:0.15 height:0.09, chair_4 x:-0.79 y:1.06 width:0.64 height:0.30, tv_5 x:0.05 y:0.32 width:0.09 height:0.13, book_6 x:0.54 y:0.62 width:0.14 height:0.09, laptop_7 x:0.04 y:0.32 width:0.09 height:0.13, book_5 x:0.53 y:0.60 width:0.14 height:0.08]

Here is the 'task description':
[A] Go 180 degrees clockwise

Here is the 'execution history' (has value if replanning):
None

Please generate the response only with a single sentence of MiniSpec program.
	"""
    llm.send_request(prompt)


import time

from jinja2.nodes import Continue

from functions_gps import *
import asyncio
import datetime
import json
import os

original_code = """tu(90);?iv('apple')==True&iv('orange'){l('Yes');->True}l('No');->False;"""
START_SAVING_URL = "http://localhost:5000/start_saving"
STOP_SAVING_URL = "http://localhost:5000/stop_saving"

async def main():
    mission_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mission_description = "liu"
    if mission_description == "":
        n = ""
    else:
        n = '_'
    mission_directory = f"mission_{mission_description}{n}{mission_timestamp}"
    with open(__file__, 'r') as f_in:
        translated_code = f_in.read()
    start_data = {
        "mission_directory": mission_directory,
        "original_code": original_code,
        "translated_code": translated_code
    }
    requests.post(START_SAVING_URL, json=start_data)
    # Notify the server to start saving images to the mission directory
    #requests.post(START_SAVING_URL, json={"mission_directory": mission_directory})

    await connect_drone()
    await ensure_armed_and_taken_off()
    #await return_to_start_position()
    # Start the print_status_text task
    #status_task = asyncio.create_task(print_status_text(drone))

    try:
        for x in range(1):
            if x % 2 == 0:
                await move_up(2)
            else:
                await move_down(2)
            #await move_down(1)
            #for _ in range(1):
            await move_forward(3)
            if x % 2 == 0:
                await turn_ccw(180)
            else:
                await turn_cw(180)
        await return_to_start_position()
    except:
        print("")
    await land_drone()
    requests.post(STOP_SAVING_URL)
    print("STOP_SAVING_URL")



if __name__ == '__main__':
    result = asyncio.run(main())
    current_datetime = datetime.datetime.now().isoformat()
    with open(__file__, 'r') as f_in:
        translated_code = f_in.read()
    log_data = {
        'date': current_datetime,
        'original_code': original_code,
        'translated_code': translated_code,
        'output': result
    }
    with open('execution_log.json', 'a') as log_file:
        log_file.write(json.dumps(log_data) + '\n')
    destination_folder = "./saved_logs"
    """os.makedirs(destination_folder, exist_ok=True)
    copy_latest_ulog(destination_folder)"""
    print("end")
    time.sleep(15)
    os._exit(0)

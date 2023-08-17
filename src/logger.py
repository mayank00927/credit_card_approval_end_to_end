import os
from datetime import datetime
from src.path_file import  log_path


def log(file_object, log_message):
    """
    function Name: log
    Description: This function logs the message with data and time:
    Output: file with message and time

    """
    print(os.getcwd())
    now = datetime.now()
    dt = now.date()
    current_time = now.strftime("%H:%M:%S")
    # os.chdir("F:\\Data Science\\Credit Card Defaulters\\log_files")
    with open(log_path+file_object, 'a') as file:
        file.write(
            str(dt) + "/" + str(current_time + "\t\t" + log_message + "\n"))


#!/bin/bash

# 
venv_activate=$(wslpath -w "/mnt/c/Users/api/apivenv/Scripts/activate.bat")
script_path=$(wslpath -w "/mnt/c/Users/api/main.py")

# Activate the virual enviroment and run the script
/mnt/c/Windows/System32/cmd.exe /c "$venv_activate && python $script_path"
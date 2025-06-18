# Find basins that do not correctly generate spin up files
# clean those basins and resubmit those basins

import os, re, glob

# Function to check if a folder contains any .nc files
def contains_nc_files(folder):
    return any(glob.glob(os.path.join(folder, "*.nc")))

# Function to remove specified files in a folder
def remove_files(folder):
    for pattern in ["*.nc", "*log*", "init_generated_files/*"]:
        for file in glob.glob(os.path.join(folder, pattern)):
            # print(file)
            os.remove(file)

            
# Function to find the range of y for a given x
def find_y_range(base_path, x):
    pattern = re.compile(f"level{x}_(\\d+)_SpinupFiles")
    y_values = set()

    for folder_name in os.listdir(base_path):
        match = pattern.match(folder_name)
        if match:
            y_values.add(int(match.group(1)))
    
    return min(y_values), max(y_values)


# Function to change directory and process 'replay.sh' file
def process_replay_script(base_folder, level_x, level_y):
    # Change directory
    cwd = os.getcwd()
    os.chdir(os.path.join(base_folder, f"level{level_x}_{level_y}"))

    # Read 'replay.sh' and extract required lines
    stop_n_line, run_startdate_line = None, None
    with open("replay.sh", "r") as file:
        for line in file:
            if line.startswith("./xmlchange STOP_N") and not stop_n_line:
                stop_n_line = line.strip()
            elif line.startswith("./xmlchange RUN_STARTDATE") and not run_startdate_line:
                run_startdate_line = line.strip()

            if stop_n_line and run_startdate_line:
                break

    # Execute extracted lines
    if stop_n_line:
        _ = os.system(stop_n_line)
    if run_startdate_line:
        _ = os.system(run_startdate_line)

    # Remove the 'spinup_info.csv' file
    if os.path.exists("spinup_info.csv"):
        _ = os.remove("spinup_info.csv")
    
    os.chdir(cwd)

# Modified function to generate the submission file
def generate_submission_file_v2(base_path, submission_file):
    with open(submission_file, "w") as f:
        for x in range(1, 4):  # Assuming range 1-9 for x
            y_min, y_max = find_y_range(base_path, x)
            print(f'For level {x}, min/max is', y_min, y_max)
            
            for y in range(y_min, y_max + 1):
                level_folder = f"level{x}_{y}_SpinupFiles"
                folder_path = os.path.join(base_path, level_folder)
                if not contains_nc_files(folder_path):
                    print(folder_path)
                    
                    # change dates
                    base_folder = "/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO"
                    process_replay_script(base_folder, x, y)
                    
                    config_file = f"level{x}-{y}_config.toml"
                    python_command = (
                        f"python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py "
                        f"/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/configuration/{config_file} SpinUp\n"
                    )
                    f.write(python_command)

                    # Remove files from the corresponding output folder
                    output_folder = f"/glade/derecho/scratch/guoqiang/CTSM_outputs/CAMELS_Calib/Calib_HH_MOASMO/level{x}_{y}/level{x}_{y}/run"
                    remove_files(output_folder)

# Define the base path
base_path = "/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO"
submission_file = os.path.join(base_path, "submission", "spinup_rerun_failed.txt")

# Generate the submission file with dynamic ranges for x and y
generate_submission_file_v2(base_path, submission_file)

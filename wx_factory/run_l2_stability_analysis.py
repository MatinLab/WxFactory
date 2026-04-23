import configparser
import subprocess
import os

ini_path = "config/"
new_folder = "stability_analysis"
ini_filename = "entropy_wave"
el_increase = [2, 4, 8]

base_output = "results"
# os.makedirs(base_output, exist_ok=True)

# 
# Create and run testcases
#

for i in el_increase:
    # Read original config
    config = configparser.ConfigParser()
    config.read(ini_path+ini_filename+".ini")
    
    print(config)
    
    # Compute new parameters
    num_elements_horizontal_new = config.getint("Spatial_discretization", "num_elements_horizontal") * i
    num_elements_vertical_new = config.getint("Spatial_discretization", "num_elements_vertical") * i
    new_filename = ini_filename + f"_h{num_elements_horizontal_new}_v{num_elements_vertical_new}"
    
    dt = config.getfloat("Time_integration", "dt")
    t_end = config.getfloat("Time_integration", "t_end")
    num_timesteps = t_end/dt


    # Modify parameters
    config["Spatial_discretization"]["num_elements_horizontal"] = str(num_elements_horizontal_new)
    config["Spatial_discretization"]["num_elements_vertical"] = str(num_elements_vertical_new)
    
    config["Output_options"]["stat_freq"] = str(0)
    config["Output_options"]["output_freq"] = str(0)
    config["Output_options"]["save_stat_freq"] = str(num_timesteps)
    config["Output_options"]["store_solver_stats"] = str(0)
    
    
    config["Output_options"]["output_dir"] = base_output + "/" + new_filename
    
    # Create temporary config file
    import os
    os.makedirs(ini_path + new_folder, exist_ok=True)
    temp_ini = ini_path + new_folder + "/" + new_filename + ".ini"
    with open(temp_ini, "w") as f:
        config.write(f)

    # Output folder
    # out_dir = f"{base_output}/nu_{nu}"
    # os.makedirs(out_dir, exist_ok=True)

    print(f"Running i = {i}")

    # Run your executable
    subprocess.run([
        "./WxFactory",
        temp_ini
    ])
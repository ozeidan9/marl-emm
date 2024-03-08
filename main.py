# %% 
# import all packages
# import ipywidgets
# print(ipywidgets.__version__)

import matplotlib.pyplot as plt


import json
from datetime import datetime

import pandas as pd
from tqdm.notebook import tqdm
from tqdm.notebook import tqdm_notebook

from flexABLE import World
import matplotlib.pyplot as plt


tqdm_notebook.pandas()
# %% 
# run training on defined scenario
study = 'storage_paper'
case = 'st_02'
run = 'run_01'
with open(f'./scenarios/{study}/{case}.json') as f:
    config = json.load(f)
    scenario = config[run]

# %%
# create world
save_policy_dir = 'policies/'

dt = scenario['dt']

snapLength = int(24/dt*scenario['days'])
timeStamps = pd.date_range(f"{scenario['year']}-01-01T00:00:00", f"{scenario['year'] + 1}-01-01T00:00:00", freq='15T')

starting_point = str(scenario['year'])+scenario['start']
starting_point = timeStamps.get_loc(starting_point)

world = World(snapshots=snapLength,
              scenario=scenario['scenario'],
              simulation_id=scenario['id'],
              starting_date=timeStamps[starting_point],
              dt=dt,
              enable_CRM=False,
              enable_DHM=False,
              check_availability=False,
              max_price=scenario['max_price'],
              write_to_db=scenario['write_to_db'],
              rl_mode=scenario['rl_mode'],
              cuda_device=0,
              save_policies=True,
              save_params={'save_dir': save_policy_dir},
              load_params=scenario['load_params'],
              learning_params = scenario['learning_params'])

# %% 
# Load scenario
world.load_scenario(startingPoint=starting_point,
                    importStorages=scenario['import_storages'],
                    opt_storages=scenario['opt_storages'],
                    importCBT=scenario['importCBT'],
                    scale=scenario['scale'])

# %% 
# Start simulation
index = pd.date_range(world.starting_date, periods=len(world.snapshots), freq=f'{str(60 * world.dt)}T')
        
if world.rl_mode and world.training:
    training_start = datetime.now()
    world.logger.info("################")
    world.logger.info(f'Training started at: {training_start}')

    learning_params = scenario['learning_params']
    training_episodes = learning_params['training_episodes']

    for i_episode in tqdm(range(training_episodes), desc='Training'):
        start = datetime.now()
        world.run_simulation()
        world.episodes_done += 1

        if ((i_episode + 1) % 5 == 0) and world.episodes_done > learning_params['learning_starts']:
            world.training = False
            world.run_simulation()
            world.compare_and_save_policies()
            world.eval_episodes_done += 1
            world.training = True

            if world.write_to_db:
                tempDF = pd.DataFrame(world.mcp, index=index, columns=['Simulation']).astype('float32')
                world.results_writer.writeDataFrame(tempDF, 'Prices', tags={'simulationID': world.simulation_id, "user": "EOM"})

    world.rl_algorithm.save_params(dir_name = 'last_policy')
    for unit in world.rl_powerplants+world.rl_storages:
        unit.save_params(dir_name = 'last_policy')

    training_end = datetime.now()
    world.logger.info("################")
    world.logger.info(f'Training time: {training_end - training_start}')
    
    world.training=False
    world.run_simulation()

    if world.write_to_db:
        world.results_writer.save_results_to_DB()
    else:
        world.results_writer.save_result_to_csv()
    
    world.logger.info("################")

else:
    start = datetime.now()
    world.run_simulation()
    end = datetime.now()
    world.logger.info(f'Simulation time: {end - start}')

    if world.write_to_db:
        world.results_writer.save_results_to_DB()
    else:
        world.results_writer.save_result_to_csv()

if world.rl_mode:
    world.logger.info(f'Average reward: {world.rl_eval_rewards[-1]}')
    world.logger.info(f'Average profit: {world.rl_eval_profits[-1]}')
    world.logger.info(f'Average regret: {world.rl_eval_regrets[-1]}')
else:
    world.logger.info(f'Average reward: {world.conv_eval_rewards[-1]}')
    world.logger.info(f'Average profit: {world.conv_eval_profits[-1]}')
    world.logger.info(f'Average regret: {world.conv_eval_regrets[-1]}')
    
# %%


import matplotlib.pyplot as plt
import pandas as pd
import os

# Define the directory path where the CSV files are stored
directory_path = 'output/storage_paper/case_02/STO_capacities/'

# Create a list of storage unit identifiers
storage_units = [f'st_{i:02}' for i in range(1, 26)]
# Initialize a DataFrame to hold all profits data
profits_df = pd.DataFrame({
    'unit': storage_units,
    'profits': [0] * len(storage_units)
})
count = 0
# Iterate over each storage unit to aggregate profits data
for unit in storage_units:
    # Construct file path for profits data
    profits_file_path = os.path.join(directory_path, f'{unit}_Profits.csv')
    
    # Check if the file exists
    if os.path.exists(profits_file_path):
        # Read the profits data from CSV file
        unit_profits = pd.read_csv(profits_file_path, index_col=0, parse_dates=True)
        # print column names and data types
        print("unit profits: ", unit_profits['Profits'].sum())
        sum = unit_profits['Profits'].sum()
        # Sum the profits for the unit and add to the DataFrame
        # append unit profits to profits_df
        
        
        profits_df['profits'][count] = sum
        count += 1

        # reset index
        # profits_df = profits_df.set_index('unit')
        # print("profits_df: ", profits_df)



# Assuming 'profits_df' has already been built up to this point as shown above
plt.figure(figsize=(14, 7))
plt.bar(profits_df['unit'], profits_df['profits'])
plt.title('Total Profits by Storage Unit')
plt.xlabel('Storage Units')
plt.ylabel('Total Profits (EUR)')
plt.xticks(rotation=90)  # Rotate x-axis labels to show them clearly
plt.tight_layout()
plt.savefig('output/storage_paper/case_02/plots/total_profits_by_unit.png')
plt.close()

print("Plot for total profits by storage unit generated.")

# %%


# Here is a python script similar to the one provided for profits. It will generate plots for total charge/discharge and state of charge for all storage units.

import matplotlib.pyplot as plt
import pandas as pd
import os

# Define the directory path where the CSV files are stored
directory_path = 'output/storage_paper/case_02/STO_capacities/'

# Create a list of storage unit identifiers
storage_units = [f'st_{i:02}' for i in range(1, 26)]
# Initialize DataFrames to hold all capacity and SOC data
capacity_df = pd.DataFrame({
    'unit': storage_units,
    'total_capacity': [0] * len(storage_units)
})
soc_df = pd.DataFrame({
    'unit': storage_units,
    'soc': [0] * len(storage_units)
})


count = 0
# Iterate over each storage unit to aggregate capacity and SOC data
for unit in storage_units:
    # Construct file paths for capacity and SOC data
    capacity_file_path = os.path.join(directory_path, f'{unit}_Capacity.csv')
    soc_file_path = os.path.join(directory_path, f'{unit}_SOC.csv')
    
    # Check if the capacity file exists and read data
    capacity_data = pd.read_csv(capacity_file_path, index_col=0, parse_dates=True)
    total_capacity = capacity_data['Total st'].sum()
    print("total_capacity: ", total_capacity)
    capacity_df['total_capacity'][count] = total_capacity

    # Check if the SOC file exists and read data
    soc_data = pd.read_csv(soc_file_path, index_col=0, parse_dates=True)
    average_soc = soc_data['SOC'].mean()  # Assuming we want the average SOC
    print("average_soc: ", average_soc)
    soc_df['soc'][count] = average_soc

    count += 1
    
    
# Plot Total Capacity
plt.figure(figsize=(14, 7))
plt.bar(capacity_df['unit'], capacity_df['total_capacity'], color='orange')
plt.title('Total Charge and Discharge Capacity by Storage Unit')
plt.xlabel('Storage Units')
plt.ylabel('Total Capacity (MWh)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('output/storage_paper/case_02/plots/total_capacity_by_unit.png')
plt.close()

# Plot State of Charge
plt.figure(figsize=(14, 7))
plt.bar(soc_df['unit'], soc_df['soc'], color='green')
plt.title('Average State of Charge (SOC) by Storage Unit')
plt.xlabel('Storage Units')
plt.ylabel('Average SOC (MWh)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('output/storage_paper/case_02/plots/average_soc_by_unit.png')
plt.close()

print("Plots for total capacity and average SOC by storage unit generated.")

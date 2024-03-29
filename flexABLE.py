# -*- coding: utf-8 -*-
"""
Copyright 2019-2020

Ramiz Qussous (INATECH - University of Freibug)
Nick Harder (INATECH - University of Freibug)
Dr. Thomas Künzel (Fichtner GmbH & Co. KG. - Hochschule Offenburg )
Prof. Dr. Anke Weidlich (INATECH - University of Freibug - Hochschule Offenburg)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the
License, or (at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# Importing classes
from agent import Agent
import EOM
import DHM
import CRM
from resultsWriter import ResultsWriter
from misc import MeritOrder, initializer
from matd3 import TD3
# from ppo import PPO 

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import os
import shutil
import time

import torch as th
from torch.utils.tensorboard import SummaryWriter

# logging level correctly
import logging

logger = logging.getLogger("")
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger('numexpr.utils').setLevel(logging.ERROR)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class World():
    """
    This is the main container
    """
    @initializer
    def __init__(self,
                 snapshots,
                 scenario="Default",
                 simulation_id=None,
                 database_name='flexRL',
                 starting_date='2018-01-01T00:00:00',
                 dt=0.25,
                 check_availability=False,
                 enable_CRM=False,
                 enable_DHM=False,
                 write_to_db=False,
                 max_price=100,
                 rl_mode=False,
                 cuda_device=0,
                 save_policies=False,
                 save_params=None,
                 load_params=None,
                 learning_params=None):

        if type(self.snapshots) == int:
            self.snapshots = list(range(snapshots))
        elif type(self.snapshots) == list:
            self.snapshots = snapshots

        self.agents = {}
        self.powerplants = []
        self.rl_powerplants = []
        self.vre_powerplants = []
        self.storages = []
        self.rl_storages = []

        self.markets = {"EOM": {},
                        "CRM": {}}

        self.currstep = 0
        self.fuelPrices = {}
        self.emissionFactors = {}

        self.minBidEOM = 1
        self.minBidCRM = 5
        self.minBidDHM = 1
        self.minBidReDIS = 1

        self.dt = dt  # Although we are always dealing with power, dt is needed to calculate the revenue and for the energy market
        self.crm_timestep = 4/dt  # The frequency of reserve market

        self.mcp = [0.]*snapshots #list of market clearing prices
        self.pfc = [0.]*snapshots #list of price forward curve
        self.IEDPrice = [2999.9]*snapshots

        self.rl_eval_rewards = []
        self.rl_eval_profits = []
        self.rl_eval_regrets = []

        self.conv_eval_rewards = []
        self.conv_eval_profits = []
        self.conv_eval_regrets = []

        self.training = self.learning_params['training']

        if self.rl_mode:
            self.obs_dim = 128
            self.act_dim = 2
            self.episodes_done = 0

            self.rl_algorithm = None

            th.backends.cuda.matmul.allow_tf32 = True
            if not self.training:
                self.device = th.device('cpu')
            else:
                cuda_device = f'cuda:{str(self.cuda_device)}'
                self.device = th.device(cuda_device if th.cuda.is_available() else 'cpu')

            self.float_type = th.float
            #self.float_type = th.float16 if self.device.type == "cuda" else th.float

            self.tensorboard_writer = None

            if self.training:
                self.learning_rate = self.learning_params['learning_rate']
                self.learning_starts = self.learning_params['learning_starts']
                self.eval_episodes_done = 0
                self.max_eval_reward = -1e9
                self.max_eval_regret = 1e9
                self.max_eval_profit = -1e9
                folder_path = 'runs/'+self.simulation_id
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    time.sleep(5)

                self.tensorboard_writer = SummaryWriter(log_dir=folder_path)

        self.logger = logger
        self.results_writer = ResultsWriter(database_name=self.database_name,
                                            simulation_id=self.simulation_id,
                                            write_to_db=write_to_db,
                                            starting_date=self.starting_date,
                                            world=self)
        
        self.current_state = None

    # def get_current_state(self):
    #     if self.rl_mode:
    #         # Use the most recent available data if the current step exceeds the data length
    #         current_prices = self.mcp[min(self.currstep, len(self.mcp) - 1)]
    #         current_demand = self.res_load_forecast['demand'][min(self.currstep, len(self.res_load_forecast) - 1)]
    #         current_load = self.res_load['demand'][min(self.currstep, len(self.res_load) - 1)]
    #         current_pfc = self.pfc[min(self.currstep, len(self.pfc) - 1)]

    #         state = np.array([current_prices, current_demand, current_load, current_pfc])

    #         # Normalize the state
    #         max_price = self.max_price
    #         max_load = max(self.res_load['demand']) if len(self.res_load['demand']) > 0 else 1

    #         self.logger.info(f'Maximum price: {max_price}, maximum load: {max_load}')
            
    #         normalized_state = state.copy()
    #         normalized_state[0] /= max_price
    #         normalized_state[1] /= max_load
    #         normalized_state[2] /= max_load
    #         normalized_state[3] /= max_price

    #         self.logger.info(f'Current state: {normalized_state}')
    #         return normalized_state

    #     else:
    #         self.logger.warning("RL mode is not enabled. Current state is not available.")
    #         return None

        
    def add_agent(self, name):
        self.agents[name] = Agent(name, world=self)


    def add_market(self, name, marketType, demand=None, CBtrades=None, HLP_DH=None, annualDemand=None):
        if marketType == "EOM":
            self.markets["EOM"][name] = EOM.EOM(
                name, demand=demand, CBtrades=CBtrades, world=self)

        if marketType == "DHM":
            self.markets["DHM"] = DHM.DHM(
                name, HLP_DH=HLP_DH, annualDemand=annualDemand, world=self)

        if marketType == "CRM":
            self.markets["CRM"] = CRM.CRM(name, demand=demand, world=self)

    def create_learning_algorithm(self):
        # if self.learning_params.get('algorithm') == 'PPO':
        #     # Initialize PPO parameters
        #     self.rl_algorithm = PPO(
        #         env=self,
        #         actor_lr=self.learning_params['actor_lr'],
        #         critic_lr=self.learning_params['critic_lr'],
        #         gamma=self.learning_params['gamma'],
        #         gae_lambda=self.learning_params['gae_lambda'],
        #         ppo_epochs=self.learning_params['ppo_epochs'],
        #         mini_batch_size=self.learning_params['mini_batch_size'],
        #         ppo_clip=self.learning_params['ppo_clip']
        #     )
        # else:
            # TD3 initialization as before
        buffer_size = int(5e5)
        self.rl_algorithm = TD3(
            env=self,
            buffer_size=buffer_size,
            learning_starts=self.learning_params['learning_starts'],
            train_freq=self.learning_params['train_freq'],
            gradient_steps=self.learning_params['gradient_steps'],
            batch_size=self.learning_params['batch_size'],
            gamma=self.learning_params['gamma']
        )

    # def create_learning_algorithm(self):
    #     buffer_size = int(5e5)
    #     self.rl_algorithm = TD3(env=self,
    #                             buffer_size=buffer_size,
    #                             learning_starts=self.learning_params['learning_starts'],
    #                             train_freq=self.learning_params['train_freq'],
    #                             gradient_steps=self.learning_params['gradient_steps'],
    #                             batch_size=self.learning_params['batch_size'],
    #                             gamma=self.learning_params['gamma'])

    def run_simulation(self):
        self.currstep = 0
        for agent in self.agents.values():
            agent.initialize()

        while self.currstep < len(self.snapshots):
            self.step()
            self.currstep += 1

        if self.rl_mode:
            # if isinstance(self.rl_algorithm, PPO):
            #     self.rl_algorithm.update_policy()  # For PPO
            # else:
            self.extract_rl_episode_info()  # For TD3

    # def run_simulation(self):
    #     self.currstep = 0
    #     for agent in self.agents.values():
    #         agent.initialize()

    #     for _ in tqdm(self.snapshots, leave=False, miniters=100, smoothing=0.1):
    #     #for _ in self.snapshots:
    #         self.step()

    #     if self.rl_mode:
    #         self.extract_rl_episode_info()
    #     else:
    #         self.extract_conv_episode_info()


    # perform a single step on each market in the following order CRM, DHM, EOM
    def step(self):
        if self.currstep < len(self.snapshots):
            if self.check_availability:
                for agent in self.agents.values():
                    agent.check_availability()

            if self.enable_CRM:
                self.markets['CRM'].step(
                    self.snapshots[self.currstep], self.agents)

            if self.enable_DHM:
                self.markets['DHM'].step(self.snapshots[self.currstep])

            for market in self.markets["EOM"].values():
                market.step(self.snapshots[self.currstep], self.agents)
            
            for agent in self.agents.values():
                agent.step()

            if self.training and self.rl_mode:
                # if isinstance(self.rl_algorithm, PPO):
                #     data = self.rl_algorithm.collect_trajectories()  # Collect data for PPO
                #     self.rl_algorithm.update_policy(data)  # Update policy with collected data
                # else:
                # TD3 specific code
                obs, actions, rewards = self.collect_experience()
                self.rl_algorithm.buffer.add(obs, actions, rewards)
                self.rl_algorithm.update_policy()

                self.current_state = obs
                
            self.currstep += 1

        else:
            self.logger.info("Reached simulation end")


    def load_scenario(self,
                      startingPoint=0,
                      importStorages=False,
                      opt_storages=False,
                      importCBT=False,
                      scale = 1):

        freq = str(60*self.dt)+'T'
        periods = len(self.snapshots)*self.dt/0.25
        index = pd.date_range(self.starting_date, periods=periods, freq='15T')

        self.logger.info("Simulation ID:{}".format(self.simulation_id))
        self.logger.info("Loading scenario: {}".format(self.scenario))

        # =====================================================================
        # Load fuel prices and emission factors
        # =====================================================================
        self.logger.info("Loading fuel data...")

        fuelData = pd.read_csv('input/{}/Fuel.csv'.format(self.scenario),
                               nrows=periods + startingPoint,
                               index_col=0)
        fuelData.drop(fuelData.index[0:startingPoint], inplace=True)
        fuelData.set_index(index, inplace=True)
        fuelData = fuelData.resample(freq).mean()
        fuelData.reset_index(drop=True, inplace=True)

        self.fuelPrices = dict(fuelData)

        emissionData = pd.read_csv(
            'input/{}/EmissionFactors.csv'.format(self.scenario), index_col=0)
        self.emissionFactors = dict(emissionData['emissions'])

        # =====================================================================
        # Create agents and load power plants
        # =====================================================================
        self.logger.info("Loading agents and assets...")

        powerplantsList = pd.read_csv('input/{}/FPP_DE.csv'.format(self.scenario),
                                      index_col=0,
                                      encoding="Latin-1")

        # =====================================================================
        # Add all unique agents (power plant operators)
        # =====================================================================
        for name in powerplantsList.company.unique():
            self.add_agent(name=name)

        # =====================================================================
        # Add availability information if provided
        # =====================================================================
        for powerplant, args in powerplantsList.iterrows():
            if args.learning == False or self.rl_mode == False:
                self.agents[args['company']].add_conv_powerplant(powerplant, **dict(args))
            else:
                self.agents[args['company']].add_rl_powerplant(powerplant, **dict(args))

        if self.load_params and self.rl_mode:
            self.logger.info("Loading RL policies from {}".format(self.load_params['id']))
        # =====================================================================
        # Adding storages
        # =====================================================================
        if importStorages:
            storages = pd.read_csv('input/{}/STO_DE.csv'.format(self.scenario),
                                      index_col=0,
                                      encoding="Latin-1")

            for _ in storages.company.unique():
                if _ not in self.agents:
                    self.add_agent(_)

            for storage, args in storages.iterrows():
                if args.rl_agent == False or self.rl_mode == False:
                    self.agents[args['company']].add_storage(storage, opt_storages, **dict(args))
                else:
                    self.agents[args['company']].add_rl_storage(storage, **dict(args))

        # =====================================================================
        # Load renewable power generation
        # =====================================================================
        vrepowerplantFeedIn = pd.read_csv('input/{}/FES_DE.csv'.format(self.scenario),
                                          index_col=0,
                                          nrows=periods + startingPoint,
                                          encoding="Latin-1")

        vrepowerplantFeedIn.drop(vrepowerplantFeedIn.index[0:startingPoint], inplace=True)
        vrepowerplantFeedIn /= scale

        vrepowerplantFeedIn.set_index(index, inplace=True)
        vrepowerplantFeedIn = vrepowerplantFeedIn.resample(freq).mean()
        vrepowerplantFeedIn.reset_index(drop=True, inplace=True)

        self.add_agent('Renewables')

        for _ in vrepowerplantFeedIn:
            self.agents['Renewables'].add_vre_powerplant(_, FeedInTimeseries=vrepowerplantFeedIn[_].to_list())

        # =====================================================================
        # Loads the inelastic demand data and cross border capacities
        # =====================================================================
        self.logger.info("Loading demand and creating EOM...")

        demand = pd.read_csv('input/{}/IED_DE.csv'.format(self.scenario),
                             nrows=periods + startingPoint,
                             index_col=0)
        demand.drop(demand.index[0:startingPoint], inplace=True)
        demand /= scale
        demand.set_index(index, inplace=True)
        demand = demand.resample(freq).mean()
        demand.reset_index(drop=True, inplace=True)

        self.res_load = demand.copy()
        self.res_load['demand'] -= vrepowerplantFeedIn.sum(axis=1)

        if importCBT:
            CBT = pd.read_csv('input/{}/CBT_DE.csv'.format(self.scenario),
                              nrows=periods + startingPoint,
                              index_col=0)
            CBT.drop(CBT.index[0:startingPoint], inplace=True)
            CBT /= scale
            CBT.set_index(index, inplace=True)
            CBT = CBT.resample(freq).mean()
            CBT.reset_index(drop=True, inplace=True)

            self.add_market('EOM_DE', 'EOM', demand=dict(
                demand['demand']), CBtrades=CBT)

            self.res_load['demand'] += CBT['Export'] - CBT['Import']

        else:
            self.add_market('EOM_DE', 'EOM', demand=dict(demand['demand']))

        # =====================================================================
        # Loads the residual load forecast
        # =====================================================================
        try:
            res_load_forecast = pd.read_csv('input/{}/RLF_DE.csv'.format(self.scenario),
                                            nrows=periods + startingPoint,
                                            index_col=0)
            res_load_forecast.drop(res_load_forecast.index[0:startingPoint], inplace=True)
            res_load_forecast /= scale
            res_load_forecast.set_index(index, inplace=True)
            res_load_forecast = res_load_forecast.resample(freq).mean()
            res_load_forecast.reset_index(drop=True, inplace=True)

            if importCBT:
                res_load_forecast['demand'] += CBT['Export'] - CBT['Import']

            self.res_load_forecast = res_load_forecast.copy()

        except:
            self.res_load_forecast = self.res_load.copy()

        # =====================================================================
        # Loads the demand for district heating
        # =====================================================================
        if self.enable_DHM:
            self.logger.info("Loading district heating demand and creating DHM...")

            HLP_DH = pd.read_csv('input/{}/HLP_DH_DE.csv'.format(self.scenario),
                                 nrows=periods + startingPoint,
                                 index_col=0)
            HLP_DH.drop(HLP_DH.index[0:startingPoint], inplace=True)
            HLP_DH.set_index(index, inplace=True)
            HLP_DH = HLP_DH.resample(freq).mean()
            HLP_DH.reset_index(drop=True, inplace=True)


            annualDemand = pd.read_csv('input/{}/DH_DE.csv'.format(self.scenario),
                                       index_col=0)
            annualDemand *= 4

            self.add_market('DHM_DE', 'DHM', HLP_DH=HLP_DH,
                           annualDemand=annualDemand)

        else:
            self.add_market('DHM_DE', 'DHM', HLP_DH=None, annualDemand=None)

        # =====================================================================
        # Loads the control reserve demand
        # =====================================================================
        if self.enable_CRM:
            self.logger.info("Loading control reserve demand and creating CRM...")

            CRM = pd.read_csv('input/{}/CRM_DE.csv'.format(self.scenario),
                              nrows=periods + startingPoint,
                              index_col=0)
            CRM.drop(CRM.index[0:startingPoint], inplace=True)
            CRM.set_index(index, inplace=True)
            CRM = CRM.resample(freq).mean()
            CRM.reset_index(drop=True, inplace=True)


            CRMdemand = {"posCRMDemand": dict(CRM['positive Demand [MW]']),
                         "negCRMDemand": dict(CRM['negative Demand [MW]']),
                         "posCRMCall": dict(CRM['positive Call-Off [MW]']),
                         "negCRMCall": dict(CRM['negative Call-Off [MW]'])}

            self.add_market('CRM_DE', 'CRM', demand=CRMdemand)

        else:
            self.add_market('CRM_DE', 'CRM', demand=None)


        self.logger.info("Calculating marginal costs...")
        for unit in self.powerplants + self.rl_powerplants:
            unit.marginal_cost = [unit.calculate_marginal_cost(t) for t in self.snapshots]

            if self.rl_mode:
                unit.scaled_marginal_cost = np.array(unit.marginal_cost).reshape(-1, 1)/self.max_price


        # =====================================================================
        # Calculate prce forward curve using simple merit order
        # =====================================================================
        self.logger.info("Calculating PFC...")

        merit_order = MeritOrder(self.res_load_forecast,
                                 self.powerplants+self.rl_powerplants,
                                 self.snapshots)

        self.pfc = merit_order.price_forward_curve()

        if self.rl_mode:
            self.logger.info("Preparing RL...")

            max_load = 70000./scale

            self.scaled_res_load_forecast = np.array(self.res_load_forecast).reshape(-1, 1)/max_load
            self.scaled_res_load = np.array(self.res_load).reshape(-1, 1)/max_load

            self.scaled_pfc = np.array(self.pfc).reshape(-1, 1)/self.max_price
            self.scaled_mcp = self.scaled_pfc.copy()

            if self.training:
                self.create_learning_algorithm()
                self.logger.info('Training mode active, MATD3 algorithm created')
                try:
                    device = th.cuda.get_device_name(device=self.device)
                except:
                    device = 'cpu'
                self.logger.info(f"Running on {device}")

        self.logger.info("All data loaded, ready to run the simulation")
        self.logger.info("################")


    def collect_experience(self):
        total_units = self.rl_algorithm.n_rl_agents
        obs = th.zeros((2, total_units, self.obs_dim), device = self.device)
        actions = th.zeros((total_units, self.act_dim), device = self.device)
        rewards = []
        
        for i, pp in enumerate(self.rl_algorithm.rl_agents):
            obs[0][i] = pp.curr_experience[0]
            obs[1][i] = pp.curr_experience[1]
            actions[i] = pp.curr_experience[2]
            rewards.append(pp.curr_experience[3])

        return  obs, actions, rewards

    def compare_and_save_policies(self):
        modes = ['reward', 'profit', 'regret']
        for mode in modes:
            value = None

            if mode == 'reward' and self.rl_eval_rewards[-1] > self.max_eval_reward:
                self.max_eval_reward = self.rl_eval_rewards[-1]
                dir_name = 'highest_reward'
                value = self.max_eval_reward
            elif mode == 'profit' and self.rl_eval_profits[-1] > self.max_eval_profit:
                self.max_eval_profit = self.rl_eval_profits[-1]
                dir_name = 'highest_profit'
                value = self.max_eval_profit
            elif mode == 'regret' and self.rl_eval_regrets[-1] < self.max_eval_regret and self.rl_eval_regrets[-1] != 0:
                self.max_eval_regret = self.rl_eval_regrets[-1]
                dir_name = 'lowest_regret'
                value = self.max_eval_regret
            
            if value is not None:
                self.rl_algorithm.save_params(dir_name=dir_name)
                for unit in self.rl_powerplants+self.rl_storages:
                    if unit.learning:
                        unit.save_params(dir_name=dir_name)

                self.logger.info(f'Policies saved, episode: {self.eval_episodes_done + 1}, mode: {mode}, value: {value:.2f}')

    def extract_rl_episode_info(self):
        total_rewards = 0
        total_profits = 0
        total_regrets = 0

        for unit in self.rl_powerplants:
            if unit.learning:
                total_rewards += sum(unit.rewards)
                total_profits += sum(unit.profits)
                total_regrets += sum(unit.regrets)

        for unit in self.rl_storages:
            if unit.learning:
                total_rewards += sum(unit.rewards)
                total_rewards += sum(unit.rewards)
                total_profits += sum(unit.profits)

        total_rl_units = self.rl_algorithm.n_rl_agents if self.rl_algorithm is not None else len(self.rl_powerplants+self.rl_storages)
        average_reward = total_rewards/total_rl_units/len(self.snapshots)
        average_profit = total_profits/total_rl_units/len(self.snapshots)
        average_regret = total_regrets/total_rl_units/len(self.snapshots)

        if self.training:
            self.tensorboard_writer.add_scalar('Train/Average Reward', average_reward, self.episodes_done)
            self.tensorboard_writer.add_scalar('Train/Average Profit', average_profit, self.episodes_done)
            self.tensorboard_writer.add_scalar('Train/Average Regret', average_regret, self.episodes_done)
        else:
            self.rl_eval_rewards.append(average_reward)
            self.rl_eval_profits.append(average_profit)
            self.rl_eval_regrets.append(average_regret)

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Eval/Average Reward', average_reward, self.eval_episodes_done)
                self.tensorboard_writer.add_scalar('Eval/Average Profit', average_profit, self.eval_episodes_done)
                self.tensorboard_writer.add_scalar('Eval/Average Regret', average_regret, self.eval_episodes_done)

    def extract_conv_episode_info(self):
        total_rewards = 0
        total_profits = 0
        total_regrets = 0

        for unit in self.powerplants:
            total_rewards += sum(unit.rewards)
            total_profits += sum(unit.profits)
            total_regrets += sum(unit.regrets)

        for unit in self.storages:
            total_rewards += sum(unit.rewards)
            total_profits += sum(unit.profits)

        total_units = len(self.powerplants+self.storages)
        self.conv_eval_rewards.append(total_rewards/total_units/len(self.snapshots))
        self.conv_eval_profits.append(total_profits/total_units/len(self.snapshots))
        self.conv_eval_regrets.append(total_regrets/total_units/len(self.snapshots))

    
# %%

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:51:41 2023

@author: nnnav
"""
import time
import numpy as np
import scipy.io as spio
from src.solver_configuration.solver_configuration import SolverConfiguration
from src import solver
import logging


def main_run(file_path,logging):
    
    try:
        np.seterr(all='raise')
        logging.info("---------Start---------")
        logging.info(f"Reading configuration from {file_path}")
        solver_config = SolverConfiguration(file_path,logging, True)
        start_time = time.time()
        p_history,u_history = solver.solver(solver_config,logging)
        mdic = {"p_history": p_history-solver_config.pressure_right,"u_history":u_history, "x_probe": solver_config.x_probe, "label": "output", "t":np.arange(0, solver_config.total_time, solver_config.dt)}
        spio.savemat('output/p_history_multi.mat', mdic)
        print("Program Completed--- %s seconds ---" % (time.time() - start_time))            
        logging.info("Program Completed--- %s seconds ---" % (time.time() - start_time))

    except FileNotFoundError as e:
        logging.error("Error in Main. File not found. Please provide the correct file path.")
        logging.error(f"{e}")
    except Exception as e:
        print(e)
        logging.error(f"An error occurred: {e}")
    
if __name__ == "__main__":
    
    # Set up logging configuration
    
    logging.basicConfig(filename='log/app.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

    #file_path = "input/train_a_2_trains.cfg"  # Update with the correct file path
    
    file_path = "input/config.cfg"  # Update with the correct file path
    #solver_config = SolverConfiguration(file_path,logging, True)
    main_run(file_path,logging) 

import argparse
from example.evaluator import PolicyGraphEvaluator
import pgeon.policy_graph as PG
from pathlib import Path
from example.environment import SelfDrivingEnvironment
from example.discretizer.discretizer_d0 import AVDiscretizer
from example.discretizer.discretizer_d1 import AVDiscretizerD1
import csv
import os

import pandas as pd

#TODO: split into 2 files: metrics computation and testing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pg_id',
                        help='The id of the Policy Graph to be loaded')
    parser.add_argument('--test_set',
                        help="csv file containg the test set of the preprocessed nuScenes dataset.")
    parser.add_argument('--policy-mode',
                        help='Whether to use the original agent, or a greedy or stochastic PG-based policy',
                        choices=['original','random','greedy', 'stochastic'])
    parser.add_argument('--action-mode',
                        help='When visiting an unknown state, whether to choose action at random or based on what similar nodes chose.',
                        choices=['random', 'similar_nodes'], default='random')     
    parser.add_argument('--discretizer', 
                        help='Specify the discretizer of the input data.', 
                        choices=['0a', '0b','0c','1a','1b','1c'], default='0a')
    parser.add_argument('--city', 
                        help='Specify city to consider when testing the PG.', 
                        choices=['all','b','s1','s2', 's3'], 
                        default="all")
    parser.add_argument('--verbose', 
                        help='Whether to make the Policy Graph code output log statements or not',
                        action='store_true')

    
    args = parser.parse_args()
    pg_id, city_id, discretizer_id, test_set, verbose = args.pg_id, args.city, args.discretizer, args.test_set, args.verbose#, args.rain, args.night
    

    #load test set
    dtype_dict = {
        'modality': 'category',  # for limited set of modalities, 'category' is efficient
        'scene_token': 'str',  
        'steering_angle': 'float64', 
        'location': 'str',
        'rain': 'int',
        'night':'int',
        'timestamp': 'str',  # to enable datetime operations
        'rotation': 'object',  # Quaternion (lists)
        'x': 'float64',
        'y': 'float64',
        'z': 'float64',
        'yaw': 'float64',  
        'velocity': 'float64',
        'acceleration': 'float64',
        'yaw_rate': 'float64'
    }
        
    #filter by city
    cities = ['boston-seaport', 'singapore-hollandvillage','singapore-onenorth','singapore-queenstown']
    if city_id == 'b': 
        city = cities[0]
    elif city_id == 's1':
        city = cities[1]
    elif city_id == 's2':
        city = cities[2]
    elif city_id == 's3':
        city = cities[3]

    test_df = pd.read_csv(Path('example/dataset/data/sets/nuscenes') / test_set, dtype=dtype_dict, parse_dates=['timestamp'])
    if city_id != 'all':
        test_df = test_df[test_df['location'] == city]
    

    if args.policy_mode == 'greedy':
        mode = PG.PGBasedPolicyMode.GREEDY
    else:
        mode = PG.PGBasedPolicyMode.STOCHASTIC

    if args.action_mode == 'random':
        node_not_found_mode = PG.PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM
    else:
        node_not_found_mode = PG.PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES
        
        
    
    nodes_path = f'example/dataset/data/policy_graphs/{pg_id}_nodes.csv'
    edges_path = f'example/dataset/data/policy_graphs/{pg_id}_edges.csv'
        
    if city_id == 'all':
        environment = SelfDrivingEnvironment(city_id)
    else:
        environment = SelfDrivingEnvironment(city)
    discretizer_configs = {
    'a': {'obj_discretizer': 'binary', 'vel_discretizer': 'binary'},
    'b': {'obj_discretizer': 'multiple', 'vel_discretizer': 'binary'},
    'c': {'obj_discretizer': 'multiple', 'vel_discretizer': 'multiple'}
    }

    default_config = {'obj_discretizer': 'multiple', 'vel_discretizer': 'multiple'}

    config = default_config
    for key in discretizer_configs:
        if key in discretizer_id:
            config = discretizer_configs[key]
            break

    DiscretizerClass = AVDiscretizer if '0' in discretizer_id else AVDiscretizerD1
    # Instantiate the discretizer with the chosen configuration
    discretizer = DiscretizerClass(
        environment,
        vel_discretization=config['vel_discretizer'],
        obj_discretization=config['obj_discretizer'],
        id=discretizer_id
    )   
    
    #load PG-based agent
    pg = PG.PolicyGraph.from_nodes_and_edges(nodes_path, edges_path, environment, discretizer)


    agent = PG.PGBasedPolicy(pg, mode, node_not_found_mode)
    
    if verbose:
        print(f'Successfully loaded Policy Graph with {len(pg.nodes)} nodes and {len(pg.edges)} edges.')
        print(f'Policy mode: {args.policy_mode}')
        print(f'Node not found mode: {node_not_found_mode}')
        print()


    
    

    if args.policy_mode == 'original':        
        output_path = 'example/results/nll.csv'
        file_exists = os.path.isfile(output_path)
        with open(output_path, 'a',newline='') as f:
            csv_w = csv.writer(f)
            if not file_exists:
                header = ['pg_id','avg_nll_action', 'std_nll_action', 'avg_nll_world', 'std_nll_world' ]
                csv_w.writerow(header)
            avg_nll_action, std_nll_action, avg_nll_world, std_nll_world = agent.compute_test_nll(test_set=test_df, verbose = verbose)
            new_row = [
                pg_id,
                avg_nll_action,
                std_nll_action, 
                avg_nll_world, 
                std_nll_world
                ]
            csv_w.writerow(new_row)
    

    
  
    
    #python3 test_pg.py --training_id pg_trainval_Cb_D0 --test_set 'test_v1.0-mini_lidar_0.csv' --policy-mode greedy --action-mode random --discretizer 0 --city 'b' --verbose
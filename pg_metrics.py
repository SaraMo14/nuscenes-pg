import argparse
from example.evaluator import PolicyGraphEvaluator
import pgeon.policy_graph as PG
from example.environment import SelfDrivingEnvironment
from example.discretizer.discretizer_d0 import AVDiscretizer
from example.discretizer.discretizer_d1 import AVDiscretizerD1
import csv
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pg_id',
                        help='The id of the Policy Graph to be loaded')
    #parser.add_argument('--verbose', 
                        #help='Whether to make the Policy Graph code output log statements or not',
                        #action='store_true')

    
    args = parser.parse_args()

    pg_id = args.pg_id       
    discretizer_id = pg_id[pg_id.find('D') + 1].split('_')[0]
    
    nodes_path = f'example/dataset/data/policy_graphs/{pg_id}_nodes.csv'
    edges_path = f'example/dataset/data/policy_graphs/{pg_id}_edges.csv'
        
    environment = SelfDrivingEnvironment(city='all')
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



    ##################
    # static metrics
    ##################
    evaluator = PolicyGraphEvaluator(pg)
    output_path = 'example/results/entropy.csv'
    file_exists = os.path.isfile(output_path)
    with open(output_path, 'a',newline='') as f:
        csv_w = csv.writer(f)
        if not file_exists:
            header = ['pg_id', 'Expected_Hs', 'Expected_Ha', 'Expected_Hw']
            csv_w.writerow(header)
        entropy_values = evaluator.compute_external_entropy()
        new_row = [
            pg_id,
            entropy_values.get('Expected_Hs'),
            entropy_values.get('Expected_Ha'),
            entropy_values.get('Expected_Hw')
            ]
        csv_w.writerow(new_row)   
  
    
    print(f'Successfully evaluated Policy Graph with {len(pg.nodes)} nodes and {len(pg.edges)} edges.')
    
    #python3 po_metrics.py --pg_id pg_trainval_Cb_D0 
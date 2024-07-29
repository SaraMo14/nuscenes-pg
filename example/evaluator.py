from pgeon.policy_graph import PolicyGraph
from collections import defaultdict
import numpy as np


class PolicyGraphEvaluator:
    def __init__(self, policy_graph: PolicyGraph):
        self.policy_graph = policy_graph
    

    def compute_internal_entropy(self):
        """
        Compute the entropy metrics for the Policy Graph.

        Returns:
        - A dictionary containing the values of H(s), Ha(s), and Hw(s) for each state.
        """
        entropy_metrics = {}
        
        for state in self.policy_graph.nodes():
            # Compute probabilities for actions given the current state
            action_freq = defaultdict(int)
            total_action_freq = 0
            
            for _, next_state, data in self.policy_graph.out_edges(state, data=True):
                action = data['action']
                freq = data['frequency'] 
                action_freq[action] += freq
                total_action_freq += freq
            
            Ha = 0
            Hw = 0
            Hs = 0
            
            for action, freq in action_freq.items():
                P_a_given_s = freq / total_action_freq #P(a|s)
                #select the edges from the policy graph that correspond to a specific action.
                action_specific_out_edges = [edge for edge in self.policy_graph.out_edges(state, data=True) if edge[2]['action'] == action]
                Ha -=P_a_given_s * np.log2(P_a_given_s)
                
                for _, next_state, data in action_specific_out_edges:
                    P_s_a_given_s = data['probability'] #p(s',a|s) 
                    Hs -=P_s_a_given_s*np.log2(P_s_a_given_s)
                    Hw -=  P_s_a_given_s* np.log2(P_s_a_given_s/P_a_given_s) #data['frequency']/freq
                    
            entropy_metrics[state] = {'p_s':self.policy_graph.nodes[state]['probability'],'Hs': Hs, 'Ha': Ha, 'Hw': Hw}
            
        return entropy_metrics
           
    
    def compute_external_entropy(self):
        '''
        Compute weighted average of internal entropy metrics across all states in the policy graph.
        
        Returns:
        - A dictionary with E[Hs], E[Ha], E[Hw].
        '''
        entropy_metrics = self.compute_internal_entropy()
        
        expected_Hs = sum(metrics['p_s'] * metrics['Hs'] for metrics in entropy_metrics.values())
        expected_Ha = sum(metrics['p_s'] * metrics['Ha'] for metrics in entropy_metrics.values())
        expected_Hw = sum(metrics['p_s'] * metrics['Hw'] for metrics in entropy_metrics.values())
        
        return {'Expected_Hs': expected_Hs, 'Expected_Ha': expected_Ha, 'Expected_Hw': expected_Hw}
    
    
    '''
    def update_state_tracking(self, state):
        """
        Updates the visited and newly discovered states based on the current state.
        """
        if state not in self.visited_states:
            self.newly_discovered_states.add(state)
        self.visited_states.add(state)


    def compute_proportion_of_discovery(self):
        """
        Computes the proportion of newly discovered states to total visited states.
        """
        if len(self.visited_states) == 0:
            return 0  # Prevent division by zero
        return len(self.newly_discovered_states) / len(self.visited_states)
    '''


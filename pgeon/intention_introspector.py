from typing import Set, List, Dict, Optional
import numpy as np
import networkx.classes.coreviews

from pgeon.policy_graph import PolicyGraph, Predicate
from pgeon.desire import Desire

class IntentionIntrospector(object):
    def __init__(self, desires: Set[Desire], pg:PolicyGraph):
        self.desires = desires
        self.pg = pg
        self.intention: Dict[Set[Predicate], Dict[Desire, float]] = {}
        self.register_all_desires( self.desires)
        
    #def __str__(self):
    #    return str(self.intention)

    def find_intentions(self, commitment_threshold: float):
        total_results = {desire.name: self.get_intention_metrics(commitment_threshold, desire) for desire in self.desires}
        return total_results


    def atom_in_state(self, node: Set[Predicate], atom: Predicate):
        return atom in node
    
    @staticmethod
    def check_state_condition(node, atom, condition_values):
        """
        Given a state (node) of predicates, checks if its predicates have the values in the condition.
        """
        for elem in node:
            if elem.predicate == atom and elem.value[0] in condition_values:
                return True
        return False

    def check_desire(self, node: Set[Predicate], desire_clause: Dict[str, Set[str]], actions_id:List[int]) -> bool:
        # Returns None if desire is not satisfied. Else, returns probability of fulfilling desire
        #   ie: executing the action when in node
        desire_clause_satisfied = True
        for atom, condition_values in desire_clause.items():
            desire_clause_satisfied = desire_clause_satisfied and self.check_state_condition(node, atom, condition_values)
            if not desire_clause_satisfied:
                return None
        return np.sum([self.get_action_probability(node, action_id) for action_id in actions_id])

   
    @staticmethod
    def get_prob(unknown_dict: Optional[Dict[str, object]]):
        if unknown_dict is None:
            return 0
        else:
            return unknown_dict.get("probability", 0)

    def get_action_probability(self, node: Set[Predicate], action_id: int):
        try:
            destinations: networkx.classes.coreviews.AdjacencyView = self.pg[node]
            return sum([self.get_prob(self.pg.get_edge_data(node, destination, key=action_id))
                        for destination in destinations])
        except KeyError:
            print(f'Warning: State {node} has no sampled successors which were asked for')
            return 0

    
    def update_intention(self, node: Set[Predicate], desire: Desire, probability: float,
                         ):
        if node not in self.intention:
            self.intention[node] = {}
        current_intention_val = self.intention[node].get(desire, 0)

        self.intention[node][desire] = current_intention_val + probability

    def propagate_intention(self, node: Set[Predicate], desire: Desire, probability,
                            stop_criterion=1e-4):
        self.update_intention(node, desire, probability)
        for coincider in self.pg.predecessors(node):
            if self.check_desire(coincider, desire.clause, desire.actions) is None:
                
                successors = self.pg.successors(coincider)
                coincider_transitions: List[Dict[Set[Predicate], float]] = \
                    [{successor: self.get_prob(self.pg.get_edge_data(coincider, successor, key=action_id)) for successor in
                      successors}
                     for action_id in self.pg.discretizer.all_actions()]
            else:
                successors = self.pg.successors(coincider)
                # If coincider can fulfill desire themselves, do not propagate it through the action_idx branch
                coincider_transitions: List[Dict[Set[Predicate], float]] = \
                    [{successor: self.get_prob(self.pg.get_edge_data(coincider, successor, key=action_id)) for successor in
                      successors}
                     for action_id in self.pg.discretizer.all_actions() if action_id not in desire.actions]

            prob_of_transition = 0
            for action_transitions in coincider_transitions:
                prob_of_transition += action_transitions.get(node, 0)

            new_coincider_intention_value = prob_of_transition * probability
            if new_coincider_intention_value >= stop_criterion:
                try:
                    coincider.propagate_intention(desire, new_coincider_intention_value)
                except RecursionError:
                    print("Maximum recursion reach, skipping branch with intention of", new_coincider_intention_value)

    def register_desire(self, desire: Desire):
        for node in self.pg.nodes:
            p = self.check_desire(node, desire.clause, desire.actions)
            if p is not None:
                self.propagate_intention(node, desire, p)

    def register_all_desires(self, desires: Set[Desire]):
        for desire in desires:
            self.register_desire(desire)

    ##############################
    # Intention and Desire metrics
    ##############################
    
    
    def get_intention_metrics(self, commitment_threshold:float, desire: Desire): 
        """
        Computes intention metrics for a specific desire or for any desire.
        """
        if desire.name != "any":
            intention_full_nodes = [node for node in self.pg.nodes if node in self.intention and desire in self.intention[node] and self.intention[node][desire]>commitment_threshold]
            node_probabilities = np.array([self.pg.nodes[node]['probability'] for node in intention_full_nodes])
            intention_probability = np.sum(node_probabilities)
            intention_vals = np.array([self.intention[node][desire] for node in intention_full_nodes])
            expected_int_probability = np.dot(intention_vals, node_probabilities)/intention_probability if intention_probability >0 else 0
        else:
            intention_full_nodes = [
                node for node in self.pg.nodes 
                if node in self.intention and 
                any(self.intention[node][d] > commitment_threshold for d in self.intention[node])]
            
            if intention_full_nodes:
                node_probabilities = np.array([self.pg.nodes[node]['probability'] for node in intention_full_nodes])
                max_intention_vals = np.array([max(self.intention[node].values()) for node in intention_full_nodes])

                intention_probability = np.sum(node_probabilities)
                expected_int_probability = np.dot(max_intention_vals, node_probabilities)/intention_probability if intention_probability >0 else 0
            else:
                intention_probability = 0
                expected_int_probability = 0

        return intention_probability, expected_int_probability


    def get_desire_metrics(self, desire):
        desire_prob, expected_action_prob = 0,0
        desire_nodes = [(node, self.check_desire(node, desire.clause, desire.actions)) for node in self.pg.nodes if self.check_desire(node,desire.clause, desire.actions) is not None]
        if desire_nodes:
            node_probabilities = np.array([self.pg.nodes[node]['probability'] for node,_ in desire_nodes])
            desire_prob = np.sum(node_probabilities)
            expected_action_prob = np.dot([p for _, p in desire_nodes],node_probabilities)/desire_prob
        return desire_prob, expected_action_prob


    def find_desires(self):
        total_results = {desire.name: self.get_desire_metrics(desire) for desire in self.desires}
        return total_results

    ##################
    # Questions
    ##################


    
    def question_intention(self, node:Set[Predicate], commitment_threshold:float):
        print(f"What do you intend to do in state s?")
        #all desires with an Id (s) over a certain threshold
        if node in self.intention:
            intented_desire = [d.name for d in self.intention[node] if self.intention[node][d] > commitment_threshold ]
            print(f'Attributed intention of the following desires: {intented_desire}')
        else:
            print("No attributed intention in this state.")

    def question6(self, desire:Desire, state:Set[Predicate]):
        print(f'How do you plan to fulfill {desire.name} from state {state}?')
        path = self.how(desire, state)
        if len(path)==0:
            print('From such state there is not path to fulfill the desire.')
        else:
            print(path) 
        
    def question5(self, desire: Desire):
        """
        Calculates the probability of performing a desirable action given the state region.
        """
        print(f"How likely are you to perform your desirable action {desire.actions} when you are in the state region {desire.clause}?")
        print(f"Probability: {self.get_desire_metrics(desire.clause)[1]}")
   
    
    def question4(self, desire: Desire):
        print(f"How likely are you to find yourself in a state where you can fulfill your desire {desire.name} by performing the action {desire.actions}?")
        print(f"Probability: {self.get_desire_metrics(desire.clause)[0]}")
     
    

    def how(self, desire:Desire, state: Set[Predicate]):
        desire_nodes = [(node, self.check_desire(node, desire.clause, desire.actions)) for node in self.pg.nodes if self.check_desire(node, desire.clause, desire.actions) is not None]
        for node, _ in desire_nodes:
            if state == node:
                return [self.pg.discretizer.get_action_from_id(action) for action in desire.actions ] #desire.actions
 
        intention_vals =  [(successor, self.intention[successor][desire]) for successor in self.pg.successors(state) if successor in self.intention and desire in self.intention[successor] ]
        if not intention_vals:
            return []
        max_successor = max(intention_vals, key=lambda x: x[1])[0]        
        actions_id = list(self.pg.get_edge_data(state, max_successor).keys())
        actions = [self.pg.discretizer.get_action_from_id(action) for action in actions_id]  #Given s, there can be 1+ actions that lead to s' 
        #NOTE: how to decide which action if there are more tha one?
        subsequent_actions = self.how(desire, max_successor)

        return [actions, max_successor] + subsequent_actions if subsequent_actions else [actions]

   

    
    
    
            
    
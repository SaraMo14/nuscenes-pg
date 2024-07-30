from typing import Set, List, Dict, Optional

import networkx.classes.coreviews

from pgeon.policy_graph import PolicyGraph, Predicate
from pgeon.desire import Desire
from pgeon.intention_aware_policy_graph import IntentionAwarePolicyGraph


class IntentionIntrospector(object):
    def __init__(self, desires: Set[Desire], pg:PolicyGraph):
        self.desires = desires
        self.pg = pg
        self.intention: Dict[Set[Predicate], Dict[Desire, float]] = {}
        
    def __str__(self):
        return str(self.intention)

    def find_intentions(self, pg: PolicyGraph, commitment_threshold: float) \
            -> IntentionAwarePolicyGraph:
        iapg = IntentionAwarePolicyGraph(pg)
        #discretizer = pg.discretizer
        #intention_full_nodes = set()
        #total_results = {k: [self.get_intention_metrics(commitment_threshold, iapg, desire)] for k in ['intention_probability', 'expected_int_probability']}
        self.register_all_desires(pg, self.desires, iapg)
        total_results = {desire.name: [self.get_intention_metrics(commitment_threshold, iapg, desire)] for desire in self.desires}
        print(total_results)
        return iapg

    def atom_in_state(self, node: Set[Predicate], atom: Predicate):
        return atom in node

    @staticmethod
    def get_prob(unknown_dict: Optional[Dict[str, object]]):
        if unknown_dict is None:
            return 0
        else:
            return unknown_dict.get("probability", 0)

    def get_action_probability(self, pg: PolicyGraph, node: Set[Predicate], action_id: int):
        try:
            destinations: networkx.classes.coreviews.AdjacencyView = pg[node]
            return sum([self.get_prob(pg.get_edge_data(node, destination, key=action_id))
                        for destination in destinations])
        except KeyError:
            print(f'Warning: State {node} has no sampled successors which were asked for')
            return 0

    def check_desire(self, pg: PolicyGraph, node: Set[Predicate], desire_clause: Set[Predicate], action_id: int):
        # Returns None if desire is not satisfied. Else, returns probability of fulfilling desire
        #   ie: executing the action when in Node
        desire_clause_satisfied = True
        for atom in desire_clause:
            desire_clause_satisfied = desire_clause_satisfied and self.atom_in_state(node, atom)
            if not desire_clause_satisfied:
                return None
        return self.get_action_probability(pg, node, action_id)

    def update_intention(self, node: Set[Predicate], desire: Desire, probability: float,
                         iapg: IntentionAwarePolicyGraph):
        if node not in iapg.intention:
            iapg.intention[node] = {}
        current_intention_val = iapg.intention[node].get(desire, 0)
        iapg.intention[node][desire] = current_intention_val + probability

    def propagate_intention(self, pg: PolicyGraph, node: Set[Predicate], desire: Desire, probability,
                            iapg: IntentionAwarePolicyGraph, stop_criterion=1e-4):
        self.update_intention(node, desire, probability, iapg)
        for coincider in pg.predecessors(node):
            if self.check_desire(pg, coincider, desire.clause, desire.action_idx) is None: #TODO: review
                successors = pg.successors(coincider)
                coincider_transitions: List[Dict[Set[Predicate], float]] = \
                    [{successor: self.get_prob(pg.get_edge_data(coincider, successor, key=action_id)) for successor in
                      successors}
                     for action_id in pg.discretizer.all_actions()]
            else:
                successors = pg.successors(coincider)
                # If coincider can fulfill desire themselves, do not propagate it through the action_idx branch
                coincider_transitions: List[Dict[Set[Predicate], float]] = \
                    [{successor: self.get_prob(pg.get_edge_data(coincider, successor, key=action_id)) for successor in
                      successors}
                     for action_id in pg.discretizer.all_actions() if action_id != desire.action_idx]

            prob_of_transition = 0
            for action_transitions in coincider_transitions:
                prob_of_transition += action_transitions.get(node, 0)
            # self.transitions = {n_idx: {action1:{dest_node1: P(dest1, action1|n_idx), ...}

            new_coincider_intention_value = prob_of_transition * probability
            if new_coincider_intention_value >= stop_criterion:
                try:
                    coincider.propagate_intention(desire, new_coincider_intention_value)
                except RecursionError:
                    print("Maximum recursion reach, skipping branch with intention of", new_coincider_intention_value)

    def register_desire(self, pg: PolicyGraph, desire: Desire, iapg: IntentionAwarePolicyGraph):
        for node in pg.nodes:
            p = self.check_desire(pg, node, desire.clause, desire.action_idx)
            if p is not None:
                self.propagate_intention(pg, node, desire, p, iapg)

    def register_all_desires(self, pg: PolicyGraph, desires: Set[Desire], iapg: IntentionAwarePolicyGraph):
        for desire in desires:
            self.register_desire(pg, desire, iapg)

    ###################
    # Intention metrics
    ###################

    def get_intention_metrics(self, commitment_threshold:float, iapg: IntentionAwarePolicyGraph, desire: Desire):
        intention_probability = 0
        expected_int_probability = 0
        for node in iapg.pg: 
            if node in iapg.intention and iapg.intention[node][desire] > commitment_threshold:
                intention_probability+=iapg.pg.nodes[node]['probability']
                expected_int_probability+=iapg.intention[node][desire]*iapg.pg.nodes[node]['probability']

        expected_int_probability = expected_int_probability / intention_probability
        return intention_probability, expected_int_probability
    
    ##################
    # Questions
    ##################
    """
    def question_intention(node:Set[Predicate], commitment_threshold:float):
        print(f"What do you intend to do in state s?")
        #all desires with an Id (s) over a certain threshold
    
    def question6(self, pg, desire:Desire, node:Set[Predicate]):
        p = self.check_desire(pg, node, desire.clause, desire.action_idx)
        if p is not node:
            pass
        print(f'How do you plan to fulfill {desire.name} from {node}?')
    """
    def check_desired_clause(self, pg: PolicyGraph, node: Set[Predicate], desire_clause: Set[Predicate]):
        # Returns None if desire clause is not in node. Else, returns probability of state.
        desire_clause_satisfied = True
        for atom in desire_clause:
            desire_clause_satisfied = desire_clause_satisfied and self.atom_in_state(node, atom)
            if not desire_clause_satisfied:
                return None
        return pg.nodes[node]['probability']
    
    def question4(self, pg:PolicyGraph, desire: Desire):
        tot_p = 0
        for node in pg.nodes:
            if self.check_desire(pg, node, desire.clause, desire.action_idx) is not None:
                #p = self.check_desired_clause(pg, node, desire.clause) #TODO:
                #tot_p+= p if p is not None else 0
                tot_p+=pg.nodes[node]['probability']
        print(f"How likely are you to find yourself in a state where you can fulfill your desire {desire.name} by performing the action {desire.action_idx}?")
        print(f"Probability: {tot_p}")
     

    def question5(self, pg:PolicyGraph, desire: Desire, action_id: int):
        """
            Calculates the probability of performing a desirable action given the state region.
        """
        p_sd = 0
        tot_p = 0
        weighted_sum = 0
        for node in pg.nodes:
            #p = self.check_desired_clause(pg, node, desire.clause) #TODO:
            #if p is not None:
            p_ad_given_s = self.check_desire(pg, node, desire.clause, desire.action_idx)
            if p_ad_given_s is not None:
                p_sd += pg.nodes[node]['probability']
                weighted_sum += p_ad_given_s * pg.nodes[node]['probability']

        tot_p = weighted_sum / p_sd

        print(f"How likely are you to perform your desirable action {desire.action_idx} when you are in the state region {desire.clause}?")
        print(f"Probability: {tot_p}")
     
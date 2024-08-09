from typing import Set, Optional, List, Dict

from pgeon.discretizer import Predicate



class Desire(object):
    def __init__(self, name: str, actions: Optional[List[int]],clause: Set[Predicate]): # clause: Dict[Predicate, List[str]]):#clause: Set[Predicate]):
        self.name = name
        self.actions = actions
        self.clause = clause # dictionary where keys are predicates that should be in Sd, and values the list of possible values they can have.


    def __repr__(self):
        return f"Desire[{self.name}]=<{self.clause}, {self.actions}>"

    def __str__(self):
        return f"Desire[{self.name}]=<{self.clause}, {self.actions}>"
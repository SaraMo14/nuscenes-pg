import random
import itertools
#from example.discretizer.utils import get_weight_assigned_color
from matplotlib import pyplot as plt
from screeninfo import get_monitors
from pyvis.network import Network
import networkx as nx



class NetworkVisualizer:
    """
    Network Visualizer.

    FOLDER_PATH (str): Folder path where the visualizations are saved.
    """

    FOLDER_PATH = './plots/graphs/'

    def __init__(self, pg, layout, name='nx'):
        """
        Constructor.

        pg: Networkx Graph
        layout: Only needed for saving the file in the corresponding layout folder.

        """
        self.pg = pg
        self.name = name
        self.layout = layout
        '''
    def get_size_nodes(self, frequencies):
        """
        Returns the size of each node in respect to the frequency in which it is visited.
        """
        minimum = 3
        maximum = 10
        node_sizes = {}

        # Compute the maximum value
        for node, actions in frequencies.items():
            partial_sum = []
            for action, next_states in actions.items():
                partial_sum.append(sum(next_states.values()))
            node_sizes[node] = sum(partial_sum)

        max_value = max(node_sizes.values())
        # Each node has size between [minimum, maximum]
        node_sizes = {node: max(minimum, (freq * maximum) // max_value) for node, freq in node_sizes.items()}

        return node_sizes

  '''

    @staticmethod
    def get_random_color():
      return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    def show(self, allow_recursion=False, font_size=7, save = False, training_id='', layout = 'circular'):
        """
        Normal plots for the graph
        """
        num_of_decimals = 2

        if layout == 'circular':
            pos = nx.circular_layout(self.pg)
        elif layout == 'spring':
            pos = nx.spring_layout(self.pg, scale = 5)
        elif layout == 'random':
            pos = nx.random_layout(self.pg)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.pg)
        elif layout == 'shell':
            pos = nx.shell_layout(self.pg)
        elif layout == 'fr':
          pos = nx.fruchterman_reingold_layout(self.pg)

        # Save the color and label of each edge
        edge_labels = {}
        edge_colors = []
        for edge in self.pg.edges:
            if edge[0] != edge[1] or allow_recursion:
                attributes = self.pg.get_edge_data(edge[0], edge[1])
                for key in attributes:
                  weight = attributes[key]['probability']
                  edge_labels[(edge[0], edge[1])] = '{} - {}'.format(
                    attributes[key]['action'],
                    round(weight, num_of_decimals)
                  )
                  edge_colors.append('#332FD0')#get_weight_assigned_color(weight))
        nodes = {node: "" for node in self.pg.nodes() #"str(node).replace('-', '\n')"
                 if self.pg.in_degree(node) + self.pg.out_degree(node) > 0}
        
        # Get node colors based on their component
        connected_components = list(nx.strongly_connected_components(self.pg))
        color_map = {}
        for component in connected_components:
            color = NetworkVisualizer.get_random_color()
            for node in component:
                
                color_map[node] = color
        node_colors = [color_map[node] for node in self.pg.nodes()]
        
        nx.draw(
            self.pg, pos,
            edge_color=edge_colors,
            width=1,
            linewidths=1,
            node_size=8,
            node_color=node_colors,
            alpha=0.8,
            arrowsize=1.5,
            labels=nodes,
            font_size=font_size,
            edgelist=[edge for edge in list(self.pg.edges()) if edge[0] != edge[1] or allow_recursion]
        )

        """
        nx.draw_networkx_edge_labels(
            self.pg, pos,
            edge_labels=edge_labels,
            font_color='#534340',
            label_pos=0.7,
            font_size=font_size
        ) """
	
        # Show the graph
        plt.show()

        # Save the Graph
        if save:
          plt.savefig(f'{training_id}.png')

    def get_file_path(self):
        """
        Returns the path where the interactive visualization will be saved.
        """
        return self.FOLDER_PATH + self.layout + '/' + self.name + '.html'

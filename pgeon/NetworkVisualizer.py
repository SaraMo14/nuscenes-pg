import random
import itertools
from example.discretizer.utils import get_assigned_color, get_weight_assigned_color
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
        options: Options for the interactive visualization.

        """
        self.pg = pg
        self.name = name
        self.layout = layout
        self.options = '''
            var options = {
    "nodes": {
    "color": {
      "highlight": {
        "border": "rgba(134,41,233,1)",
        "background": "rgba(198,171,255,1)"
      }
    },
    "font": {
      "size": 25,
      "strokeWidth": 5
    }
  },
  "edges": {
    "arrowStrikethrough": false,
    "color": {
      "inherit": true,
      "highlight": {
        "border": "rgba(57,70,233,1)"
      }
    },
    "font": {
      "strokeWidth": 10
    },
    "selfReferenceSize": 50,
    "smooth": {
      "forceDirection": "none"
    }
  },
  "interaction": {
    "hover": false,
    "multiselect": true
  },
  "physics": {
    "barnesHut": {
      "damping": 1,
      "avoidOverlap": 0.9,
      "gravitationalConstant": -23000
    },
    "minVelocity": 0.75
  }
}
        '''

    def get_size_nodes(self, frequencies):
        """
        Returns the size of each node in respect to the frequency in which it is visited.
        """
        minimum = 30
        maximum = 100
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

    def get_legend_nodes(self, width, height):
        """
        Saves the edges legend in legend.html
        """
        # Num of possible actions
        num_legend_nodes = 6
        step = 100

        # Center of the screen
        x = 0
        y = -(height / 2) + step
        colors = ['#' + get_assigned_color(action) for action in range(6)]
        legend_nodes = [
            (
                len(self.pg.nodes) + legend_node,
                {
                    'color': colors[legend_node],
                    'label': legend_node,#action_num_to_str(legend_node),
                    'size': 30,
                    # 'fixed': True, # So that we can move the legend nodes around to arrange them better
                    'physics': False,
                    'x': x,
                    'y': f'{y + legend_node * step}px',

                    'widthConstraint': 50,
                    'font': {'size': 20}
                }
            )
            for legend_node in range(num_legend_nodes)
        ]

        # Building Graph
        nt = Network(height=f"{height}px", width=f"{width}px", directed=True)
        g = nx.Graph()
        g.add_nodes_from(legend_nodes)
        nt.from_nx(g)
        nt.toggle_physics(True)
        nt.set_edge_smooth('dynamic')

        # Save the html file
        nt.show(self.FOLDER_PATH + 'legend.html')

    def show_interactive(self, frequencies, show_options=False, second_display=False, subgraph=None):
        """
        Saves the MDP in a html file for the interactive visualization.
        """
        node_sizes = self.get_size_nodes(frequencies)
        nx.set_node_attributes(self.pg, node_sizes, 'size')

        width = 500
        height = 500
        proportions = (0.63, 0.78)
        if not show_options:
            proportions = (0.98, 0.78)

        # Getting the screen size to show the graph
        try:
            for m in get_monitors():
                if not second_display and m.is_primary or second_display and not m.is_primary:
                    width = m.width * proportions[0]
                    height = m.height * proportions[1]
        except:
            pass

        # Displaying graph
        nt = Network(height=f"{height}px", width=f"{width}px", directed=True)


        if subgraph is not None:
            n = 0
            for node in list(self.pg.nodes()):
                num = len(self.pg.edges(node))
                if num >= subgraph[0] and num <= subgraph[1] and random.randint(0,4) == 2:
                    n = node
                    break
            s = 0
            subnodes = [n]
            for u, v, w in set(self.pg.edges(n, data='probability')):
                s += w
                subnodes.append(v)

            self.pg = self.pg.subgraph(subnodes)
            edges = []
            for node in self.pg.edges(data=True):
                if node[0] == n or node[1] == n and (node[0], node[1]) not in edges:
                    node[2]['label'] = round(node[2]['probability'] ,3)
                    edges.append((node[0], node[1]))


        nt.from_nx(self.pg)
        nt.toggle_physics(False)
        nt.set_edge_smooth('dynamic')
        self.get_legend_nodes(700 ,700)


        # Show options if requested
        if show_options:
            nt.show_buttons()
        else:
            nt.set_options(self.options)

        # Save the html file
        nt.show(self.get_file_path())

    @staticmethod
    def get_random_color():
      return "#{:06x}".format(random.randint(0, 0xFFFFFF))


    def show(self, allow_recursion=False, font_size=7, save = False, training_id=''):
        """
        Normal plots for the graph
        """
        num_of_decimals = 2

        pos = nx.circular_layout(self.pg)

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
                  edge_colors.append(get_weight_assigned_color(weight))
        nodes = {node: "" for node in self.pg.nodes() #"str(node).replace('-', '\n')"
                 if self.pg.in_degree(node) + self.pg.out_degree(node) > 0}
        
        # Get node colors based on their component
        connected_components = list(nx.weakly_connected_components(self.pg))
        connected_components = list(nx.weakly_connected_components(self.pg))
        color_map = {}
        for component in connected_components:
            color = NetworkVisualizer.get_random_color()
            for node in component:
                color_map[node] = color
        node_colors = [color_map[node] for node in self.pg.nodes()]
        
        nx.draw(
            self.pg, #pos,
            edge_color=edge_colors,
            width=3,
            linewidths=1,
            node_size=100,
            node_color=node_colors,
            alpha=0.8,
            arrowsize=15,
            labels=nodes,
            font_size=font_size,
            edgelist=[edge for edge in list(self.pg.edges()) if edge[0] != edge[1] or allow_recursion]
        )

        nx.draw_networkx_edge_labels(
            self.pg, pos,
            edge_labels=edge_labels,
            font_color='#534340',
            label_pos=0.7,
            font_size=font_size
        )

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

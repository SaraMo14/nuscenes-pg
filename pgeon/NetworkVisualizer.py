import random
from matplotlib import pyplot as plt
import networkx as nx



class NetworkVisualizer:

    FOLDER_PATH = './plots/graphs/' #Folder path where the visualizations are saved.

    def __init__(self, pg, layout, name='nx'):
        """
        Constructor.

        pg: Networkx Graph
        layout: Only needed for saving the file in the corresponding layout folder.

        """
        self.pg = pg
        self.name = name
        self.layout = layout


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
                  edge_colors.append('#332FD0')
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

 
        if save:
          plt.savefig(f'{training_id}.png')
        else:
            plt.show()

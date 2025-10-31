import numpy as np
import matplotlib.pyplot as plt

# edges in form [start_edge, stop_edge)
# assume imput arrays are sorted

class Bin:
    def __init__(self, data, weights, start_edge, stop_edge, indices):
        if len(data) == 0:
            self.count = 0
            self.data = []
            self.indices = []
            self.start_edge = start_edge
            self.stop_edge = stop_edge
        else:
            self.data = data
            self.start_edge = start_edge
            self.stop_edge = stop_edge
            self.indices = indices

            self.mean = np.mean(self.data)
            self.count = 0
            for i in range(len(self.data)):
                self.count += weights[i]

    def create_sub_histogram(self, uniform, width_or_edges, data, weights, indices):
        self.sub_histogram = Histogram(uniform = uniform)
        sorted_indexes = np.argsort(data)
        sort_data = np.take_along_axis(data, sorted_indexes, axis = -1)
        sort_weights = np.take_along_axis(weights, sorted_indexes, axis = -1)
        sort_indices = np.take_along_axis(indices, sorted_indexes, axis = -1)
        self.sub_histogram.fill_histogram(width_or_edges, sort_data, sort_weights, sort_indices)
    

class Histogram:
    def __init__(self, uniform, bins = False, edges = False):
        self.uniform = uniform
        if bins:
            self.bins = bins
        else:
            self.bins = []
        if edges:
            self.edges = edges
        else:
            self.edges = []
    
    #adds on rightmost edge of bin
    def add_bin(self, Bin):
        self.bins.append(Bin)

    def fill_histogram(self, width_or_edges, data, weights, indices):
        if self.uniform:
            self.width = width_or_edges
            self.edges.append(data[0])
            last_point_index = 0
            next_edge = self.edges[-1] + self.width
            for i in range(len(data)):
                if data[i] >= next_edge:
                    this_bin = Bin(data[last_point_index:i], weights[last_point_index:i], self.edges[-1], next_edge, indices[last_point_index:i])
                    self.add_bin(this_bin)
                    self.edges.append(next_edge)
                    next_edge = self.edges[-1] + self.width

                    #Check to see if there are empty bins and if so, add them
                    if data[i] - self.edges[-1] >= self.width:
                        empty_bins = int((data[i] - self.edges[-1]) / self.width)
                        for _ in range(empty_bins):
                            empty_bin = Bin([], [], self.edges[-1], next_edge, [])
                            self.add_bin(empty_bin)
                            self.edges.append(next_edge)
                            next_edge = self.edges[-1] + self.width
                    
                    last_point_index = i

                    if next_edge > data[-1]:
                        last_bin = Bin(data[last_point_index:], weights[last_point_index:], self.edges[-1], next_edge, indices[last_point_index:])
                        self.add_bin(last_bin)
                        self.edges.append(next_edge)
                        break
        else:
            self.edges = width_or_edges
            last_point_index = 0
            for i in range(len(data)):
                if data[i] >= self.edges[0]:
                    last_point_index = i
                    break
            next_edge_index = 1
            for i in range(len(data)):
                if data[i] >= self.edges[next_edge_index]:
                    this_bin = Bin(data[last_point_index:i], weights[last_point_index:i], self.edges[next_edge_index - 1], self.edges[next_edge_index], indices[last_point_index:i])
                    self.add_bin(this_bin)
                    next_edge_index += 1

                    #handle empty bins
                    while data[i] >= self.edges[next_edge_index]:
                        empty_bin = Bin([], [], self.edges[next_edge_index - 1], self.edges[next_edge_index], [])
                        self.add_bin(empty_bin)
                        next_edge_index += 1
                    
                    last_point_index = i

                    if self.edges[next_edge_index] > data[-1]:
                        last_bin = Bin(data[last_point_index:], weights[last_point_index:], self.edges[next_edge_index - 1], self.edges[next_edge_index], indices[last_point_index:])
                        self.add_bin(last_bin)
                        break
    
    def graph_it(self, color, label):
        counts = []
        for b in self.bins:
            counts.append(b.count)
        plt.bar(self.edges[0:-1], counts, width=(self.width if self.uniform else np.diff(self.edges)), align='edge', edgecolor = 'black', color=color, alpha=0.5)
        plt.xlabel(label)
        plt.ylabel('Count (Weighted)')
        plt.show()

    def get_point_data(self):
        data_points = []
        for b in self.bins:
            data_points.append(len(b.data))
        #min
        min_point = np.min(data_points)
        #max
        max_point = np.max(data_points)
        #mean
        mean_point = np.mean(data_points)
        return [min_point, max_point, mean_point]
    
    def get_bin_data(self):
        counts = []
        for b in self.bins:
            counts.append(b.count)
        #mean counts
        mean_count = np.mean(counts)
        #min count
        min_count = np.min(counts)
        #max count
        max_count = np.max(counts)
        return [min_count, max_count, mean_count]

    def get_counts(self):
        counts = []
        for b in self.bins:
            counts.append(b.count)
        return np.array(counts)

    def statistics(self):
        #number of bins
        num_bins = len(self.bins)
        #number of data points in bins
        point_data = self.get_point_data()
        #counts with weights
        bin_data = self.get_bin_data()
        string = f'''
        BIN WIDTH: {self.width if self.uniform else 'NOT UNIFORM'} \n 
        Number of Bins: {num_bins} \n
        Minimum Statistics in Bin: {point_data[0]} \n 
        Maximum Statistics in Bin: {point_data[1]} \n
        Mean Number of Statistics in Bin: {point_data[2]} \n 
        Minimum value of bin after weights: {bin_data[0]} \n 
        Maximum value of bin after weights: {bin_data[1]} \n
        Mean value of bin after weights: {bin_data[2]} 
        '''
        return string
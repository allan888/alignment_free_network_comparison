__author__= 'weiliangxing'

"""
NOTE:
1. in order to ensure working properly, you need to run once generate
a feature vector for a graph one time; then begin experiment
2. if you use normalized vector for training, when you test a new feature vector.
is that necessary to also normalize it? if so, normalize with whose other vectors?

"""

import threading
import numpy.linalg
import networkx as nx
from matplotlib import pylab as pl
import random
import timeit
import sys

dir = ""
type = ""
# single feature vector generation class
# note: not normalized.
class GenerateFeatures:
    def __init__(self, graph, feature_list=[]):
        self.no_feature = 39
        self.G = graph
        self.nodes = nx.number_of_nodes(self.G)
        self.edges = nx.number_of_edges(self.G)
        self.Lap = nx.normalized_laplacian_matrix(self.G)
        # ??? how to check whether comparable, addable?
        self.eigvals = numpy.linalg.eigvals(self.Lap.A).tolist()
        try:
            self.radius = nx.radius(self.G)
        except nx.exception.NetworkXError:
            self.radius = "ND"
        try:
            self.ecc_dic = nx.eccentricity(self.G)
        except nx.exception.NetworkXError:
            self.ecc_dic = {}
        self.degree_dic = nx.average_neighbor_degree(self.G)
        self.pagerank = nx.pagerank(self.G).values()
        if feature_list == []:
            self.feature_list = list(range(1, self.no_feature + 1))
        else:
            self.feature_list = feature_list
        self.feature_vector = []
        self.feature_time = []

    # function to call all feature functions efficiently
    def build_vector(self):
        for i in self.feature_list:
            method = getattr(self, 'f' + str(i), lambda: "f1")
            f = method()
            # print("finish feature %s" % i)
            self.feature_vector.append(f)

        # purely for experiment purpose
        self.write_single_vector()

    # for experiment purpose
    # optional for writing files for label 3
    def write_single_vector(self):
        label = 4
        with open(dir+"/output.txt", "a") as output:
                output.write("%s\n" % label)
                output.write("%s\n" % self.feature_vector)
        print("write done for single vector.")
        # with open(dir+"/output_time.txt", "a") as output:
        #         output.write("%s\n" % label)
        #         output.write("%s\n" % self.feature_time)
        # print("write done for single vector's time.")

    # number of nodes:
    def f1(self):
        # start = timeit.default_timer()
        start = 0
        n = self.nodes
        # stop = timeit.default_timer()
        stop = 0
        # self.feature_time.append(stop - start)
        return n

    # number of edges:
    def f2(self):
        start = 0
        e = self.edges
        stop = 0
        # self.feature_time.append(stop - start)
        return e

    # average clustering coefficient
    def f3(self):
        start = 0
        c = nx.average_clustering(self.G)
        stop = 0
        # self.feature_time.append(stop - start)
        return c

    # average degree
    def f4(self):
        start = 0
        degree_list = nx.degree(self.G).values()
        total = sum(degree_list)
        res = total/len(degree_list)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # average neighbor degree
    def f5(self):
        start = 0
        total = sum(self.degree_dic.values())
        res = total/len(self.degree_dic)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # average path length:
    def f6(self):
        start = 0
        len_dic = nx.closeness_centrality(self.G)
        total = sum(len_dic.values())
        res = total/len(len_dic)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # energy:
    def f7(self):
        start = 0
        sum_of_squares = sum([i ** 2 for i in self.eigvals])
        stop = 0
        # self.feature_time.append(stop - start)
        return sum_of_squares

    # periphery:
    def f8(self):
        return "ND"
        start = 0
        try:
            per_list = nx.periphery(self.G)
        except nx.exception.NetworkXError:
            per_list = {}

        if per_list != {}:
            res = len(per_list)/self.nodes
        else:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # diameter: !!! note: define at least 90% while here 100%
    def f9(self):
        return "ND"
        start = 0
        try:
            res = nx.diameter(self.G)
        except nx.exception.NetworkXError:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # radius: !!! note: define at least 90% while here 100%
    def f10(self):
        return "ND"
        start = 0
        res = self.radius
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # percentage of central points:
    def f11(self):
        return "ND"
        start = 0
        if self.ecc_dic != {}:
            cent_p = 0
            for key, value in self.ecc_dic.items():
                if value == self.radius:
                    cent_p += 1
        else:
            cent_p = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return cent_p

    # average effective eccentricity:
    # !!! note: define at least 90% while here 100%
    def f12(self):
        return "ND"
        start = 0
        if self.ecc_dic != {}:
            total = sum(self.ecc_dic.values())
            res = total/len(self.ecc_dic)
        else:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # transitivity:
    def f13(self):
        start = 0
        res = nx.transitivity(self.G)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # number of eigenvalue
    def f14(self):
        start = 0
        res = len(self.eigvals)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # second largest eigenvalue
    def f15(self):
        start = 0
        try:
            m = max(self.eigvals)
            new_m = max(n for n in self.eigvals if n!= m)
        except TypeError:
            new_m = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return new_m

    # spectral radius
    # !!! note: sometimes max(self.eigvals) will generate errors
    # saying complex > complex is not comparable. You may need to run
    # it again.
    def f16(self):
        start = 0
        try:
            res = max(self.eigvals)
        except TypeError:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # giant connected ratio:
    def f17(self):
        start = 0
        total = self.nodes
        giant_graph = max(nx.connected_component_subgraphs(self.G), key=len)
        no_giant = nx.number_of_nodes(giant_graph)
        res = no_giant/total
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # trace
    def f18(self):
        start = 0
        try:
            res = sum(self.eigvals)
        except TypeError:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # percentage of isolated points:
    def f19(self):
        start = 0

        zero_p = 0
        for key, value in self.degree_dic.items():
            if value == 0:
                zero_p += 1
        total = self.nodes
        res = zero_p/total
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # percentage of end points
    def f20(self):
        start = 0
        one_p = 0
        for key, value in self.degree_dic.items():
            if value == 1:
                one_p += 1
        total = self.nodes
        res = one_p/total
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # number of maximal cliques
    def f21(self):
        start = 0
        clique_list = list(nx.find_cliques(self.G))
        res = len(clique_list)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # cycle basis:
    def f22(self):
        start = 0
        cycle_list = nx.cycle_basis(self.G)
        res = len(cycle_list)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # square clustering coefficient:
    def f23(self):
        start = 0

        square_dic = nx.square_clustering(self.G)
        total = sum(square_dic.values())
        no = len(square_dic.values())
        res = total/no
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # Degree assortativity coefficient
    def f24(self):
        start = 0
        res = nx.degree_assortativity_coefficient(self.G)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # entropy:
    # !!! note: not label here, using central distr instead
    def f25(self):
        return "ND"
        start = 0
        c_vals = nx.degree_centrality(self.G).values()
        norm_c = []
        total = sum(c_vals)
        for i in c_vals:
            norm_c.append(i/total)
        norm_c = numpy.asarray(norm_c)
        if norm_c.all() != 0:
            entropy = numpy.nansum(norm_c * numpy.log2(1/norm_c))
        else:
            entropy = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return entropy

    # !!! additional features: by weiliang xing
    # density
    def f26(self):
        start = 0
        res = nx.density(self.G)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # total shortest-path betweeness centrality for nodes
    def f27(self):
        start = 0
        c_vals = nx.betweenness_centrality(self.G).values()
        res = sum(c_vals)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # total shortest-path betweeness centrality for edges
    def f28(self):
        start = 0
        c_vals = nx.edge_betweenness_centrality(self.G).values()
        res = sum(c_vals)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # average current-flow closeness centrality for nodes
    # information centrality
    def f29(self):
        return "ND"
        start = 0
        try:
            c_vals = nx.current_flow_closeness_centrality(self.G).values()
            total = len(c_vals)
            res = sum(c_vals) / total
        except nx.NetworkXError:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # current-flow betweenness centrality for nodes
    def f30(self):
        return "ND"
        start = 0
        try:
            c_vals = nx.current_flow_betweenness_centrality(self.G).values()
            res = sum(c_vals)
        except nx.NetworkXError:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # current-flow betweenness centrality for edges
    def f31(self):
        return "ND"
        start = 0
        try:
            c_vals = nx.edge_current_flow_betweenness_centrality(self.G).values()
            res = sum(c_vals)
        except nx.NetworkXError:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # Compute the eigenvector centrality for the graph G
    def f32(self):
        start = 0
        try:
            c_vals = nx.eigenvector_centrality(self.G).values()
            res = sum(c_vals)
        except nx.exception.NetworkXError:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # The center is the set of nodes with eccentricity equal to radius
    def f33(self):
        return "ND"
        start = 0
        try:
            res = len(nx.center(self.G))
        except nx.exception.NetworkXError:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # total pagerank
    def f34(self):
        start = 0
        res = sum(self.pagerank)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # average pagerank
    def f35(self):
        start = 0
        total = len(self.pagerank)
        res = sum(self.pagerank) / total
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # sum of load centrality for nodes
    def f36(self):
        start = 0
        s = nx.load_centrality(self.G).values()
        res = sum(s)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # Find a maximal cardinality matching in the graph.
    def f37(self):
        start = 0
        s = nx.maximal_matching(self.G)
        res = len(s)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # Return a random maximal independent set guaranteed to
    # contain a given set of nodes.
    def f38(self):
        start = 0
        try:
            d = nx.maximal_independent_set(self.G)
            res = len(d)
        except nx.exception.NetworkXUnfeasible:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # average shortest path length
    def f39(self):
        return "ND"
        start = 0
        try:
            res = nx.average_shortest_path_length(self.G)
        except nx.exception.NetworkXError:
            res = "ND"
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # average closeness vitality for nodes of graph G
    def f40(self):
        start = 0
        v = nx.closeness_vitality(self.G).values()
        res = sum(v) / len(v)
        stop = 0
        # self.feature_time.append(stop - start)
        return res

    # def f00(self):
    # min_g = nx.minimum_spanning_tree(self.G)

    # undone yet
    # Eigen-exponent: undone
    # Hop-plot exponent: undone


# class to generate a feature vector for a graph
class FeatureVectorGenerator:
    def __init__(self, graph, feature_list=[]):
        self.graph = graph
        self.feature_list = feature_list
        self.feature_vector = []
        self.vector_range_normal = []
        self.vector_z_normal = []
        self.gen_feature_vector()

    def gen_feature_vector(self):
        f = GenerateFeatures(self.graph, self.feature_list)
        f.build_vector()
        self.feature_vector = f.feature_vector

    def gen_range_normal_vector(self, min_l, max_l):
        # assume every vector has same length
        for i in range(len(self.feature_vector)):
            if self.feature_vector[i] < min_l[i]:
                min_l[i] = self.feature_vector[i]
            if self.feature_vector[i] > max_l[i]:
                max_l[i] = self.feature_vector[i]
            diff = max_l[i] - min_l[i]
            if diff == 0:
                r = 0
            else:
                r = (self.feature_vector[i] - min_l[i]) / diff
            self.vector_range_normal.append(r)

    def gen_z_normal_vector(self, mean_l, std_l):
        # assume every vector has same length
        for i in range(len(self.feature_vector)):
            if std_l[i] == 0:
                r = 0
            else:
                r = (self.feature_vector[i] - mean_l[i]) / std_l[i]
                self.vector_z_normal.append(r)


# class to generate feature vectors for all input graph list
# without normalizations
class FeatureVectorsGenerator:
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
        self.feature_vectors = []
        self.gen_feature_vectors()

    def gen_feature_vectors(self):
        for i in range(len(self.graphs)):
            graph = self.graphs[i]
            f = GenerateFeatures(graph)
            f.build_vector()
            self.feature_vectors.append(f.feature_vector)
            print("feature gen done for graph %s" % i)


# class to generate feature vectors with normalizations
# note this class is for whole complete vectors.
class FeatureNormalizationGenerator:
    def __init__(self, feature_vectors):
        self.feature_vectors = feature_vectors
        self.min = []
        self.max = []
        self.mean = []
        self.std = []
        self.vectors_range_normal = [[] for i in range(len(self.feature_vectors))]
        self.vectors_z_normal = [[] for i in range(len(self.feature_vectors))]
        self.get_parameter()

    def get_parameter(self):
        feature_arrays = numpy.asarray(self.feature_vectors)
        self.min = feature_arrays.min(axis=0).tolist()
        self.max = feature_arrays.max(axis=0).tolist()
        self.mean = feature_arrays.mean(axis=0).tolist()
        self.std = feature_arrays.std(axis=0).tolist()

    # !!! note: Here rule: if max-min == 0, then return 0
    # r(x) = (x - min)/ (max - min)
    def gen_vector_range_normal(self):
        for i in range(len(self.feature_vectors)):
            for j in range(len(self.feature_vectors[i])):
                diff = self.max[j] - self.min[j]
                if diff == 0:
                    r = 0
                else:
                    r = (self.feature_vectors[i][j] - self.min[j]) / diff
                self.vectors_range_normal[i].append(r)

    # !!! note: Here rule: if std == 0, then return 0
    # z(x) = (x - mean)/std
    def gen_vector_z_normal(self):
        for i in range(len(self.feature_vectors)):
            for j in range(len(self.feature_vectors[i])):
                if self.std[j] == 0:
                    r = 0
                else:
                    r = (self.feature_vectors[i][j] - self.mean[j]) / self.std[j]
                self.vectors_z_normal[i].append(r)

    def write_vector_normal(self, out_dir):
        pass


# !!! note: write file by append mode
class WriteToFile:
    def __init__(self, feature_vectors, labels):
        self.vectors = feature_vectors
        self.labels = labels

    def write_by_append(self, out_dir):
        with open(dir+"/"+out_dir, "a") as output:
            for i in range(len(self.labels)):
                output.write("%s\n" % self.labels[i])
                output.write("%s\n" % self.vectors[i])
        print("write done.")


# class for concurrent processing of large amount of graphs
# for un-normalized vectors
class ProcessGraphs(threading.Thread):
    """
    Threads for same size of graphs list concurrency process
    """
    count_thread = 0

    def __init__(self, graphs, labels, lock, out_dir):
        threading.Thread.__init__(self)
        ProcessGraphs.count_thread += 1
        self.graphs = graphs
        self.labels = labels
        self.lock = lock
        self.out_dir = out_dir
        self.vectors = []

    def run(self):
        try:
            generator = FeatureVectorsGenerator(self.graphs, self.labels)
            self.vectors = generator.feature_vectors
        except:
            print("Errors in feature generation occur in thread %s" % ProcessGraphs.count_thread)
            raise
        print("PreProcessing of group %s finished" % self.labels[0])

        self.lock.acquire()
        print("lock acquired by label %s" % self.labels[0])
        write_obj = WriteToFile(self.vectors, self.labels)
        write_obj.write_by_append(self.out_dir)
        print("lock released by label %s" % self.labels[0])
        self.lock.release()


# class GenerateGraphs exists ONLY for tests purpose
# simulation to generate many graphs
# format input would be generated BY this class:
# a list of object graph, a list of graphs' label.
# label 1 : "PAM"
class GenerateGraphs:
    def __init__(self, label):
        g1 = GenerateGraph("PAM", 30, 10)
        g2 = GenerateGraph("PAM", 30, 15)
        g3 = GenerateGraph("PAM", 30, 20)
        self.graphs = [g1, g2, g3]
        self.labels = [label, label, label]


# class to generate single graph for tests purpose
# simulation to generate a graph
# only support PAM and ER family.
class GenerateGraph:
    def __init__(self, name, para1, para2):
        if name == "PAM":
            # nodes, edges, and nodes > edges
            self.G = nx.barabasi_albert_graph(para1, para2)
            # nodes, prob
        elif name == "ER":
            self.G = nx.erdos_renyi_graph(para1, para2)

    # def export_graph(self, out_file):

    def draw_figure(self):
        g = self.G
        pos = nx.spring_layout(g)
        pl.figure(1)
        pl.title("original figure with family {f}".format(f="first"))
        nx.draw(g, pos=pos)
        # nx.draw_networkx_labels(g, pos, font_size=10, font_family='sans-serif')
        # pl.savefig("origin_with_prob_{p}.png".format(p=probability))
        pl.show()


class GenGraphs:
    def __init__(self, nodes, rho, seed, rewire):
        self.nodes = nodes
        self.density = rho
        self.edges = int(rho * nodes * (nodes - 1) / 2)
        self.avg_deg = int(2 * self.edges / self.nodes)
        self.seed = seed
        self.rewire = rewire

        self.graphs = []
        self.graphs_rewire = []
        self.labels = []
        self.labels_rewire = []

    def gen_pam(self):
        for i in range(1, self.seed + 1):
            g = nx.barabasi_albert_graph(self.nodes, self.avg_deg, seed=i)
            for j in range(10, self.rewire + 1, 10):
                num_rewire = int(j*self.edges/100)
                g_tmp = nx.double_edge_swap(g, num_rewire, self.edges)
                self.graphs_rewire.append(g_tmp)
                self.labels_rewire.append(1)
            self.graphs.append(g)
            self.labels.append(1)

    def gen_er(self):
        for i in range(1, self.seed + 1):
            g = nx.erdos_renyi_graph(self.nodes, self.density, seed=i, directed=False)
            for j in range(10, self.rewire + 1, 10):
                num_rewire = int(j * self.edges / 100)
                g_tmp = nx.double_edge_swap(g, num_rewire, self.edges)
                self.graphs_rewire.append(g_tmp)
                self.labels_rewire.append(2)
            self.graphs.append(g)
            self.labels.append(2)

    def gen_geo(self):
        for i in range(1, self.seed + 1):
            g = nx.random_geometric_graph(self.nodes, 0.042)
            for j in range(10, self.rewire + 1, 10):
                num_rewire = int(j * self.edges / 100)
                g_tmp = nx.double_edge_swap(g, num_rewire, self.edges)
                self.graphs_rewire.append(g_tmp)
                self.labels_rewire.append(3)
            self.graphs.append(g)
            self.labels.append(3)

    def gen_ddm(self):
        for i in range(1, self.seed + 1):
            g = self.duplication_divergence_model(0.5)
            for j in range(10, self.rewire + 1, 10):
                num_rewire = int(j * self.edges / 100)
                g_tmp = nx.double_edge_swap(g, num_rewire, self.edges)
                self.graphs_rewire.append(g_tmp)
                self.labels_rewire.append(4)
            self.graphs.append(g)
            self.labels.append(4)

    def duplication_divergence_model(self, sigma):
        node_set = set(range(0, self.nodes))
        g = nx.Graph()
        g.add_node(0)
        node_set.remove(0)

        while len(g.nodes()) < self.nodes:
            u = random.randint(0, len(g.nodes()))
            v = list(node_set)[0]
            node_set.remove(v)
            g.add_node(v)
            if len(g.nodes()) == 2:
                g.add_edge(u, v)
            else:
                for w in g.neighbors(u):
                    if random.random() < sigma:
                        g.add_edge(v, w)
                if len(g.neighbors(v)) == 0:
                    g.add_edge(u, v)
        return g

def gen(type, nodes, rho, seed, rewire, need_rewire):
    g1 = GenGraphs(nodes, rho, seed, rewire)

    if type == "ddm":
        g1.gen_ddm()
    elif type == "geo":
        g1.gen_geo()
    elif type == "er":
        g1.gen_er()
    elif type == "pam":
        g1.gen_pam()
    else:
        print("type error")
        return -1

    print("graphs generation done")

    if need_rewire:
        generator = FeatureVectorsGenerator(g1.graphs_rewire, g1.labels)
        print("feature generation done")
        write_obj = WriteToFile(generator.feature_vectors, generator.labels)
        write_obj.write_by_append(str(nodes)+"_"+str(rho)+"_"+str(seed)+"_"+str(rewire)+".txt")
    else:
        generator = FeatureVectorsGenerator(g1.graphs, g1.labels)
        print("feature generation done")
        write_obj = WriteToFile(generator.feature_vectors, generator.labels)
        write_obj.write_by_append(str(nodes)+"_"+str(rho)+"_"+str(seed)+"_"+str(rewire)+".txt")

    return 0

def run_exp(type):

    # =======feature vector development======
    # g1 = GenerateGraph("PAM", 30, 10)
    # # f_list = [1, 2, 4, 5]
    # f_list = []
    # start = timeit.default_timer()
    # v1 = FeatureVectorGenerator(g1.G, f_list)
    # print(v1.feature_vector)
    # stop = timeit.default_timer()
    # print(stop - start)

    # =======feature development=======
    # g1 = GenerateGraph("PAM", 30, 10)
    # # v1 = GenerateFeatures(g1.G, [1, 2, 4, 5])
    # v1 = GenerateFeatures(g1.G)
    # v1.build_vector()
    # print(v1.feature_vector)
    # print(v1.f00())

    # =======single thread sample========
    #
    # vectors_generator = GenerateGraphs(0)
    # generator = FeatureVectorsGenerator(vectors_generator.graphs, vectors_generator.labels)
    # print("original feature vectors:")
    # for i in generator.feature_vectors:
    #     print(i)
    # normal_vectors =FeatureNormalizationGenerator(generator.feature_vectors)
    # normal_vectors.gen_vector_range_normal()
    # normal_vectors.gen_vector_z_normal()
    # print("range_normalization:")
    # for m in normal_vectors.vectors_range_normal:
    #     print(m)
    # print("z-normalization:")
    # for m in normal_vectors.vectors_z_normal:
    #     print(m)
    # write_obj = WriteToFile(generator.feature_vectors, generator.labels)
    # write_obj.write_by_append("ouput.txt")

    # ========concurrency sample========
    # graphs0 = GenerateGraphs(0)
    # graphs1 = GenerateGraphs(1)
    # lock = threading.Lock()
    # t1 = ProcessGraphs(graphs0.graphs, graphs0.labels, lock)
    # t2 = ProcessGraphs(graphs1.graphs, graphs1.labels, lock)
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()

    # ==== test for graphs gen======
    # FAIL
    # DO NOT TRY THIS
    # g0 = GenGraphs(1000, 0.01, 10, 10)
    # g0.gen_ddm()
    # print(len(g0.graphs))
    # print(len(g0.graphs_rewire))
    # print(g0.labels[0])

    # g1 = GenGraphs(1000, 0.01, 10, 10)
    # g2 = GenGraphs(1000, 0.01, 10, 10)
    # g3 = GenGraphs(1000, 0.01, 10, 10)
    # g4 = GenGraphs(1000, 0.01, 10, 10)
    # g1.gen_pam()
    # g2.gen_er()
    # g3.gen_geo()
    # g4.gen_ddm()
    # print("done for graphs generation")
    # lock = threading.Lock()
    # t1 = ProcessGraphs(g1.graphs, g1.labels, lock, "output.txt")
    # t2 = ProcessGraphs(g2.graphs, g2.labels, lock, "output.txt")
    # t3 = ProcessGraphs(g3.graphs, g3.labels, lock, "output.txt")
    # t4 = ProcessGraphs(g4.graphs, g4.labels, lock, "output.txt")
    #
    # t1.start()
    # t2.start()
    # t3.start()
    # t4.start()
    # t1.join()
    # t2.join()
    # t3.join()
    # t4.join()


    paras = {500:[0.01,0.02,0.05],800:[0.01],900:[0.03],1100:[0.01],1200:[0.015],1000:[0.01,0.02],1500:[0.01,0.02]}
    
    phase = 1
    for nodes in paras:
        for rho in paras[nodes]:
            if gen(type,nodes,rho,20,10,True) != 0:
                print("error in generation process")
                return -1
            if gen(type,nodes,rho,20,10,False) != 0:
                print("error in generation process")
                return -1
            print("done phase "+str(phase))
            phase += 1
    print("ddm done")


if __name__ == "__main__":
    if len (sys.argv) < 3:
        print ( "Usage: python generate_feature.py [pam|ddm|er|geo] [rewire|normal]")
    else:
        type  = sys.argv[1]
        rewire = sys.argv[2]
        if not type in ["pam","er","ddm","geo"]:
            print("error type")
        elif not rewire in ["rewire","normal"]:
            print("error rewire specification")
        else:
            dir = type+"_"+rewire
            run_exp(type)

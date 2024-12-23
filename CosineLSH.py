import numpy as np
import pickle
import math


class LSH:
    class HashTable:
        def __init__(self, num_hyperplanes, inp_dimensions, sim_threshold):
            self.num_hyperplanes = num_hyperplanes
            self.inp_dimensions = inp_dimensions
            self.sim_threshold = sim_threshold
            self.hash_table = dict()
            self.projections = np.random.randn(self.num_hyperplanes, self.inp_dimensions)

        def generate_hash(self, inp_vector):
            bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
            return ''.join(bools.astype('str'))

        def __setitem__(self, inp_vec, key):
            hash_value = self.generate_hash(inp_vec)
            self.hash_table[hash_value] = self.hash_table \
                                              .get(hash_value, list()) + [{"s": key, "v": inp_vec}]

        def __getitem__(self, inp_vec):
            hash_value = self.generate_hash(inp_vec)
            recs = self.hash_table.get(hash_value, [])
            refined_vals = dict()

            for r in recs:
                sim = self.cosine_similarity(inp_vec, r["v"])
                if sim >= self.sim_threshold:
                    refined_vals.setdefault(r["s"], sim)  # r["v"])
            return refined_vals

        def cosine_similarity(self, v1, v2):
            return np.dot(v1, v2)  # /(np.linalg.norm(v1) * np.linalg.norm(v2))  we have normalized vectors

    def __init__(self, num_hyperplanes, inp_dimensions, sim_threshold, success_prob):
        self.num_hyperplanes = num_hyperplanes
        self.inp_dimensions = inp_dimensions
        self.sim_threshold = sim_threshold

        self.success_prob = success_prob
        # Compute the angle corresponding to the similarity threshold
        theta = math.acos(sim_threshold)
        # Compute the collision probability for a single hash function
        P = 1 - theta / math.pi
        # Compute the number of hash tables needed
        self.num_tables = math.ceil(math.log(1 - self.success_prob) / math.log(1 - P ** self.num_hyperplanes))

        self.num_tables = 50

        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(self.HashTable(self.num_hyperplanes, self.inp_dimensions, self.sim_threshold))
        self.vs = {}
        print("L=", self.num_tables)

    def get_vs(self, key):
        return self.vs[key]

    def save(self, fileName):
        file = open(fileName, "wb")
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def load(fileName):
        file = open(fileName, "rb")
        return (pickle.load(file))

    # key is the plain blocking key
    def __setitem__(self, key, inp_vec, label):
        nv = self.normalize(inp_vec)
        for table in self.hash_tables:
            table[nv] = key  # label
        self.vs[key] = {"v": label, "h": nv}

    def __getitem__(self, inp_vec):
        # results = list()
        results = []  # dict()
        matchingKeys = {}
        nv = self.normalize(inp_vec)
        n = 0
        for table in self.hash_tables:
            results1 = table[nv]
            for r in results1:
                n += 1
                key = r
                if key in matchingKeys.keys():
                    continue
                results.append({"k": key, "v": self.vs[key]["v"]})
                matchingKeys[key] = 1
            # results = { **results , **table[inp_vec] }

        # return list(set(results))
        return results, n

    def normalize(self, v):
        norm = np.linalg.norm(v)
        # Normalize the vector
        if norm != 0:  # Avoid division by zero
            nv = v / norm
        else:
            nv = v  # Keep as is if the norm is zero
        return nv

    def add(self, key, v, s):
        self.__setitem__(key, v, s)

    def get(self, v):
        return self.__getitem__(v)


'''
v1 = [0.03, 0.6, 0.7]
v2 = [0.031, 0.5, 0.7]
lsh = LSH(15, 2, 3)
lsh.hash(v1, "s1")
lsh.hash(v2, "s2")
q1 = [0.03, 0.6, 0.7]
print(lsh.query(q1))
'''

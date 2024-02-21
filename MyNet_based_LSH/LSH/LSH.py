from sklearn.metrics.pairwise import euclidean_distances

class LSH:
    def __init__(self, data, hash_size=10):
        self.data = data
        self.hash_size = hash_size
        self.hash_tables = self.build_hash_tables()

    def build_hash_tables(self):
        hash_tables = [{} for _ in range(self.hash_size)]
        for i, point in enumerate(self.data):
            hashes = self.hash_function(point)  # 得到每个数据的 hash 值
            for table_index, hash_value in enumerate(hashes):
                if hash_value in hash_tables[table_index]:
                    hash_tables[table_index][hash_value].append(i)  # i 是 data 索引
                else:
                    hash_tables[table_index][hash_value] = [i]
        return hash_tables

    def hash_function(self, point):
        # signature签名阵, 所以他的索引是 tables 的索引
        return [hash(tuple(point)) % 100 for _ in range(self.hash_size)]

    def query(self, query_point, num_candidates=5):
        # 取候选
        query_hashes = self.hash_function(query_point)
        candidate_set = set()
        for table_index, hash_value in enumerate(query_hashes):
            if hash_value in self.hash_tables[table_index]:
                candidate_set.update(self.hash_tables[table_index][hash_value])
        candidates = list(candidate_set)


        if len(candidates) == 0:
            return []
        
        # 计算与候选的距离
        distances = euclidean_distances([query_point], self.data[candidates])[0]

        # 排序，根据距离排序
        sorted_candidates = sorted(zip(candidates, distances), key=lambda x: x[1])
        return [(i[0], list(self.data[i[0]]), i[1]) for i in sorted_candidates][: num_candidates]
        # return sorted_candidates[:num_candidates]


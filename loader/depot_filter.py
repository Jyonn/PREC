from UniTok import UniDep


class DepotFilter(UniDep):
    def __init__(self, store_dir):
        super(DepotFilter, self).__init__(store_dir=store_dir)

    def remove_empty(self, col):
        valid_sample_indexes = []

        for sample in self:
            if sample[col]:
                valid_sample_indexes.append(sample[self.id_col])
        self.index_order = valid_sample_indexes
        self.sample_size = len(self.index_order)

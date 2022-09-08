import pandas as pd
from UniTok.tok import SplitTok, EntTok


class PredictTok(SplitTok):
    def t(self, obj):
        predicts = []
        if pd.notnull(obj):
            objs = obj.split(self.sep)
            for o in objs:
                nid, click = o.split('-')
                predicts.append([self.vocab.append(nid), int(click)])
        return predicts

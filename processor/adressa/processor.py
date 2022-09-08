import collections
import datetime
import json
import os.path
import random

import numpy as np
import pandas as pd
from UniTok import UniTok, Column, Vocab, UniDep, Plot
from UniTok.column import IndexColumn
from UniTok.tok import IdTok, BertTok, SplitTok, EntTok, BaseTok
from tqdm import tqdm

from processor.processor import Processor


class HistoryTok(BaseTok):
    def t(self, obj: list):
        return [self.vocab.append(o) for o in obj]


class Depot:
    def __init__(self, *keys):
        self.data = dict()
        self.keys = keys
        for k in keys:
            self.data[k] = []

    def append(self, **kwargs):
        for k in kwargs:
            assert k in self.keys
            self.data[k].append(kwargs[k])

    def migrate(self):
        return pd.DataFrame(self.data)


class AdressaProcessor(Processor):
    def __init__(self, data_dir, store_dir):
        super().__init__(data_dir=data_dir, store_dir=store_dir)
        self.content_dir = os.path.join(self.data_dir, 'Content')

        self.start_date = datetime.datetime(year=2017, month=1, day=1)
        self.file_list = self.get_file_list()
        self.news_set = set()
        self.user_set = set()
        self.news_depot = Depot(
            'nid', 'title', 'keywords', 'publishTime', 'url',
            'entities', 'categories', 'locations'
        )
        self.user_depot = Depot(
            'uid', 'os', 'country', 'region', 'city', 'deviceType',
        )
        self.inter_depot = Depot(
            'uid', 'nid', 'visitTime', 'activeTime', 'newsAge',
        )

        self.news_csv = os.path.join(self.store_dir, 'news.csv')
        self.user_csv = os.path.join(self.store_dir, 'user.csv')
        self.inter_csv = os.path.join(self.store_dir, 'inter.csv')
        self.content_news_csv = os.path.join(self.store_dir, 'content-news.csv')

        self.news_store_path = os.path.join(self.store_dir, 'news')

    def get_file_list(self):
        days = 90
        start_date = self.start_date
        file_list = []
        for _ in range(days):
            file_list.append(start_date.strftime('%Y%m%d'))
            start_date += datetime.timedelta(days=1)
        return file_list

    @staticmethod
    def parse_profile(profile):
        locations, entities, categories = [], [], []
        if profile:
            for attr in profile:
                for group in attr['groups']:
                    if group['weight'] > 0.9:
                        if group['group'] == 'location':
                            locations.append(attr['item'])
                        elif group['group'] == 'entity':
                            entities.append(attr['item'])
                        elif group['group'] == 'category':
                            categories.append(attr['item'])
        return '^'.join(locations), '^'.join(entities), '^'.join(categories)

    def parse_line(self, line):
        if line[-1] == '\n':
            line = line[:-1]

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return

        attr_required = ['id', 'userId', 'title']
        for attr in attr_required:
            if attr not in data:
                return

        uid, nid = data['userId'], data['id']

        active_time, visit_time, publish_time = data.get('activeTime'), data['time'], data.get('publishtime')

        visit_time -= 7 * 60 * 60

        if publish_time:
            publish_time = datetime.datetime.strptime(publish_time, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()

        if nid not in self.news_set:
            self.news_set.add(nid)

            title, keywords, url = data.get('title'), data.get('keywords'), data.get('canonicalUrl')
            profile = data.get('profile')
            locations, entities, categories = self.parse_profile(profile)

            self.news_depot.append(
                nid=nid,
                title=title,
                keywords=keywords,
                publishTime=publish_time,
                url=url,
                entities=entities,
                categories=categories,
                locations=locations,
            )

        if uid not in self.user_set:
            self.user_set.add(uid)

            country, region, city = data.get('country'), data.get('region'), data.get('city')
            device_type, system = data.get('deviceType'), data.get('os')

            self.user_depot.append(
                uid=uid,
                country=country,
                region=region,
                city=city,
                deviceType=device_type,
                os=system,
            )

        self.inter_depot.append(
            uid=uid,
            nid=nid,
            activeTime=active_time,
            visitTime=visit_time,
            newsAge=visit_time - publish_time if publish_time else None
        )

    def parse_file(self, filepath):
        with open(filepath, 'r') as f:
            for line in tqdm(f):
                self.parse_line(line)

    def parse_data(self):
        for file in tqdm(self.file_list):
            self.parse_file(os.path.join(self.data_dir, file))

        # user_df = self.user_depot.migrate()
        # news_df = self.news_depot.migrate()
        inter_df = self.inter_depot.migrate()

        # user_df.to_csv(self.user_csv, index=False, sep='\t')
        # news_df.to_csv(self.news_csv, index=False, sep='\t')
        inter_df.to_csv(self.inter_csv, index=False, sep='\t')

    def append_news_content(self):
        df = pd.read_csv(self.news_csv, sep='\t')

        descriptions = []
        publish_times = []
        for item in df.itertuples():
            publish_time = item.publishTime
            try:
                data = json.load(open(os.path.join(self.content_dir, item.nid), 'r'))
            except Exception:
                descriptions.append(None)
                publish_times.append(publish_time)
                continue

            description = None
            for field in data['fields']:
                if field['field'] == 'publishtime':
                    publish_time = datetime.datetime.strptime(field['value'], '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
                elif field['field'] == 'description':
                    description = field['value']

            publish_times.append(publish_time)
            descriptions.append(description)

        df['publishTime'] = publish_times
        df['description'] = descriptions
        df.to_csv(self.content_news_csv, sep='\t', index=False)

    @staticmethod
    def build_inter_ut(nid_vocab: Vocab, uid_vocab: Vocab):
        return UniTok().add_col(IndexColumn(
            name='index'
        )).add_col(Column(
            name='uid',
            tokenizer=EntTok(name='uid', vocab=uid_vocab).as_sing(),
        )).add_col(Column(
            name='nid',
            tokenizer=EntTok(name='nid', vocab=nid_vocab).as_sing(),
        )).add_col(Column(
            name='click',
            tokenizer=EntTok(name='click').as_sing(),
        ))

    @staticmethod
    def build_user_ut(nid_vocab: Vocab, history_length):
        user_ut = UniTok().add_col(Column(
            name='uid',
            tokenizer=IdTok(name='uid').as_sing(),
        )).add_col(Column(
            name='device',
            tokenizer=EntTok(name='device').as_sing(),
        )).add_col(Column(
            name='os',
            tokenizer=EntTok(name='os').as_sing(),
        ))

        for col in ["country", "region", "city"]:
            vocab = Vocab(name=col).reserve(["[OOV]"])
            user_ut.add_col(Column(
                name=col,
                tokenizer=EntTok(name=col, vocab=vocab).as_sing()
            ))

        user_ut.add_col(Column(
            name='history',
            tokenizer=HistoryTok(
                name='history',
                vocab=nid_vocab
            ).as_list(
                max_length=history_length,
                slice_post=True
            ),
        ))
        return user_ut

    @staticmethod
    def build_news_ut():
        lang_tok = BertTok('language', 'pretrained/norwegian_bert_uncased')
        entity_vocab = Vocab('entity')
        entity_tok = SplitTok(
            name='entities',
            sep='^',
            vocab=entity_vocab
        )

        return UniTok().add_col(Column(
            name='nid',
            tokenizer=IdTok(name='nid').as_sing(),
        )).add_col(Column(
            name='title',
            tokenizer=lang_tok.as_list(max_length=20),
        )).add_col(Column(
            name='desc',
            tokenizer=lang_tok.as_list(max_length=50),
        )).add_col(Column(
            name='key',
            tokenizer=SplitTok(
                name='keywords',
                sep=',',
                vocab=entity_vocab
            ).as_list(max_length=8),
        )).add_col(Column(
            name='ent',
            tokenizer=entity_tok.as_list(max_length=4),
        )).add_col(Column(
            name='cat',
            tokenizer=entity_tok.as_list(max_length=3),
        )).add_col(Column(
            name='loc',
            tokenizer=entity_tok.as_list(max_length=3),
        ))

    def read_inter_file(self):
        return pd.read_csv(
            filepath_or_buffer=self.inter_csv,
            sep='\t',
            header=0,
            names=['uid', 'nid', 'time', 'active', 'age'],
            usecols=['uid', 'nid', 'time']
        )

    def read_user_file(self):
        return pd.read_csv(
            filepath_or_buffer=self.user_csv,
            sep='\t',
            header=0,
            names=['uid', 'os', 'country', 'region', 'city', 'device']
        )

    def read_news_file(self):
        return pd.read_csv(
            filepath_or_buffer=self.content_news_csv,
            sep='\t',
            header=0,
            names=['nid', 'title', 'key', 'time', 'url', 'ent', 'cat', 'loc', 'desc']
        )

    def tokenize_news(self):
        news_ut = self.build_news_ut()
        news_df = self.read_news_file()
        news_ut.read_file(news_df).analyse()

        entities_vocab = news_ut.vocab_depot.get_vocab('entity')  # type: Vocab
        entities_vocab.trim_vocab(min_frequency=5)
        news_ut.analyse().tokenize().store_data(self.news_store_path)

    def parse_days(self, days):
        if isinstance(days, int):
            days = [days, days + 1]
        start_date = self.start_date + datetime.timedelta(days=days[0])
        end_date = self.start_date + datetime.timedelta(days=days[1])
        print(start_date, end_date)
        return [start_date.timestamp(), end_date.timestamp()]

    @staticmethod
    def filter(df: pd.DataFrame, days):
        return df[(days[1] > df.time) & (df.time >= days[0])]

    def build_history(self, df: pd.DataFrame, history_days):
        history_df = self.filter(df, history_days)
        groups = history_df.groupby('uid')

        user_history = dict()
        for g in tqdm(groups):
            group = g[1]  # type: pd.DataFrame
            group = group.sort_values('time')
            user_history[g[0]] = group.nid.to_list()

        return user_history

    def tokenize_user(self, history_days, user_path, history_length):
        history_days = self.parse_days(history_days)
        # train_days = self.parse_days(train_days)
        # dev_days = self.parse_days(dev_days)

        news_depot = UniDep(self.news_store_path)
        nid_vocab = news_depot.vocab_depot.get_vocab('nid')
        user_ut = self.build_user_ut(nid_vocab, history_length)

        user_df = self.read_user_file()
        inter_df = self.read_inter_file()  # type: pd.DataFrame

        user_history_dict = self.build_history(inter_df, history_days)
        user_history = []
        for uid in user_df.uid:
            if uid in user_history_dict:
                user_history.append(user_history_dict[uid])
            else:
                user_history.append([])

        user_df['history'] = user_history
        user_ut.read_file(user_df).analyse()

        country_vocab = user_ut.vocab_depot('country')
        region_vocab = user_ut.vocab_depot('region')
        city_vocab = user_ut.vocab_depot('city')
        country_vocab.trim_svocab(min_frequency=5, oov_default=0)
        region_vocab.trim_vocab(min_frequency=5, oov_default=0)
        city_vocab.trim_vocab(min_frequency=5, oov_default=0)

        user_ut.tokenize().store_data(os.path.join(self.store_dir, user_path))

    @staticmethod
    def get_neg_samples(user_dict, uid, news_depot: UniDep, total_count):
        nids = []
        nid_vocab = news_depot.vocab_depot.get_vocab(news_depot.id_vocab)

        while total_count:
            index = random.randint(0, news_depot.sample_size - 1)
            nid = nid_vocab.index2obj[index]
            if 'uid' not in user_dict or nid not in user_dict[uid]:
                nids.append(nid)
                total_count -= 1
        return nids

    def tokenize_inter(self, history_days, train_days, dev_days, user_path, inter_path, neg_ratio, dev_test_ratio):
        news_depot = UniDep(self.news_store_path)
        nid_vocab = news_depot.vocab_depot.get_vocab('nid')
        user_depot = UniDep(os.path.join(self.store_dir, user_path))
        uid_vocab = user_depot.vocab_depot.get_vocab('uid')
        inter_ut = self.build_inter_ut(nid_vocab=nid_vocab, uid_vocab=uid_vocab)

        history_days = self.parse_days(history_days)
        train_days = self.parse_days(train_days)
        dev_days = self.parse_days(dev_days)

        assert history_days[1] == train_days[0]
        history_days[1] = train_days[1]

        inter_df = self.read_inter_file()  # type: pd.DataFrame

        user_dict = self.build_history(inter_df, history_days)
        train_df = self.filter(inter_df, train_days)
        dev_and_test_df = self.filter(inter_df, dev_days)
        groups = dev_and_test_df.groupby('uid')
        dev_dfs, test_dfs = [], []
        for g in groups:
            if random.random() < dev_test_ratio:
                dev_dfs.append(g[1])
            else:
                test_dfs.append(g[1])
        dev_df = pd.concat(dev_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        for df, mode in zip([train_df, dev_df, test_df], ['train', 'dev', 'test']):
            print(mode, len(df))
            del df['time']
            df['click'] = [1] * len(df)
            groups = df.groupby('uid')
            nid_list = []
            uid_list = []
            for g in groups:
                nids = self.get_neg_samples(user_dict, g[0], news_depot, len(g[1]) * neg_ratio)
                nid_list.extend(nids)
                uid_list.extend([g[0]] * len(nids))
            neg_df = pd.DataFrame(dict(nid=nid_list, uid=uid_list, click=[0] * len(nid_list)))
            df = pd.concat([df, neg_df], ignore_index=True)

            inter_ut.read_file(df).tokenize().store_data(os.path.join(self.store_dir, inter_path, mode))

    def build_fuxi_dataset(self, user_path, inter_path, user_test_path=None, modes=None):
        news_depot = UniDep(self.news_store_path)
        user_depot = UniDep(os.path.join(self.store_dir, user_path))
        train_depot = UniDep(os.path.join(self.store_dir, inter_path, 'train'))
        dev_depot = UniDep(os.path.join(self.store_dir, inter_path, 'dev'))
        test_depot = UniDep(os.path.join(self.store_dir, inter_path, 'test'))

        if not user_test_path:
            user_test_depot = user_depot
        else:
            user_test_depot = UniDep(os.path.join(self.store_dir, user_test_path))

        attr_format = lambda tokens: '^'.join(map(str, tokens))

        for inter_depot, mode in zip([train_depot, dev_depot, test_depot], ['train', 'dev', 'test']):
            if modes and mode not in modes:
                continue

            f = open(os.path.join(self.store_dir, 'fuxi-{}.csv'.format(mode)), 'w+')
            f.write(self._to_fuxi_line())
            for sample in tqdm(inter_depot):
                if mode in ['dev', 'test']:
                    user_info = user_test_depot[sample['uid']]
                else:
                    user_info = user_depot[sample['uid']]
                news_info = news_depot[sample['nid']]
                sample['imp'] = sample['uid']

                sample.update(user_info)
                sample.update(news_info)

                for attr in ['title', 'desc', 'key', 'cat', 'loc', 'ent', 'history']:
                    sample[attr] = attr_format(sample[attr])
                f.write(self._to_fuxi_line(sample))
            f.close()

    @staticmethod
    def _to_fuxi_line(sample=None):
        attrs = ['imp', 'uid', 'nid', 'title', 'desc', 'key', 'ent', 'cat', 'loc', 'device', 'os', 'country', 'region', 'city', 'click', 'history']

        if not sample:
            return '\t'.join(attrs) + '\n'

        s = []
        for attr in attrs:
            s.append(str(sample[attr]))
        return '\t'.join(s) + '\n'

    def build_gnud_dataset(self, user_path, inter_path, store_path):
        news_depot = UniDep(self.news_store_path)
        user_depot = UniDep(os.path.join(self.store_dir, user_path))

        store_path = os.path.join(self.store_dir, store_path)
        os.makedirs(store_path, exist_ok=True)

        user_vocab = Vocab(name='user')
        user_vocab.reserve(['[PAD]'])

        data = dict()
        for mode in ['train', 'dev', 'test']:
            depot = UniDep(os.path.join(self.store_dir, inter_path, mode))
            inter_data = []
            for user_sample in depot:
                inter_data.append([
                    user_vocab.append(user_sample['uid']),
                    user_sample['nid'],
                    0,
                    user_sample['click'],
                ])
            data[f'{mode}_data'] = np.array(inter_data)

        news_vocab = Vocab(name='news')
        news_vocab.reserve(['[PAD]'])
        news_entity = []
        news_group = []
        news_title = []
        for news_sample in tqdm(news_depot):
            group = []
            entity = []
            title = []
            for col_id, col in enumerate(['key', 'ent', 'cat', 'loc']):
                group.extend([col_id + 1] * len(news_sample[col]))
                for token in news_sample[col]:
                    entity.append(news_vocab.append(f'{col}_{token}'))
            news_group.append(group + [0] * (40 - len(group)))
            news_entity.append(entity + [0] * (40 - len(entity)))
            for token in news_sample['title']:
                title.append(news_vocab.append(f'tit_{token}'))
            news_title.append(title + [0] * (40 - len(title)))

        data['news_entity'] = np.array(news_entity)
        data['news_group'] = np.array(news_group)
        data['news_title'] = np.array(news_title)

        np.save(os.path.join(store_path, 'data.npy'), data)
        user_vocab.save(store_path)
        news_vocab.save(store_path)

        user_news = [[]]
        user_news_dict = dict()
        news_user = dict()
        for user_sample in user_depot:
            if user_sample['uid'] not in user_vocab.obj2index:
                continue
            uid = user_vocab.append(user_sample['uid'])
            history = user_sample['history']
            if not history:
                continue
            user_news_dict[uid] = history
            for nid in history:
                if nid not in news_user:
                    news_user[nid] = []
                news_user[nid].append(uid)

        for i in range(user_vocab.get_size()):
            if not i:
                continue
            if i in user_news_dict:
                user_news.append(user_news_dict[i])
            else:
                user_news.append([])

        for nid in news_user:
            random.shuffle(news_user[nid])
            news_user[nid] = news_user[nid][:50]

        json.dump(user_news, open(os.path.join(store_path, 'user_news.json'), 'w'))
        json.dump(news_user, open(os.path.join(store_path, 'news_user.json'), 'w'))

    def tokenize(self):
        pass

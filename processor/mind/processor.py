import json
import os.path
import random

import numpy as np
import pandas as pd
import requests
from UniTok import UniTok, Column, Vocab, UniDep
from UniTok.tok import IdTok, EntTok, BertTok, SplitTok
from bs4 import BeautifulSoup as Souper
from tqdm import tqdm

from processor.processor import Processor
from processor.mind.tok.predict_tok import PredictTok


class MindProcessor(Processor):
    def __init__(self, data_dir, store_dir, body_path=None):
        super(MindProcessor, self).__init__(data_dir=data_dir, store_dir=store_dir)

        self.body_path = body_path
        self.news_store_dir = os.path.join(self.store_dir, 'news')
        self.user_store_dir = os.path.join(self.store_dir, 'user')
        self.user_test_store_dir = os.path.join(self.store_dir, 'user-test')
        self.train_store_dir = os.path.join(self.store_dir, 'train')
        self.dev_store_dir = os.path.join(self.store_dir, 'dev')
        self.test_store_dir = os.path.join(self.store_dir, 'test')

    def read_news_data(self, mode):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, mode + '/news.tsv'),
            sep='\t',
            names=['nid', 'cat', 'subCat', 'title', 'abs', 'url', 'titEnt', 'absEnt'],
            usecols=['nid', 'cat', 'subCat', 'title', 'abs'],
        )

    def read_body_data(self):
        bodies = dict()
        with open(self.body_path, 'r') as f:
            for line in f:
                body_data = json.loads(line)
                bodies[body_data['nid']] = body_data['body']
        return bodies

    def read_user_data(self, mode):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, mode + '/behaviors.tsv'),
            sep='\t',
            names=['session', 'uid', 'time', 'history', 'predict'],
            usecols=['uid', 'history']
        )

    def read_inter_data(self, mode):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, mode + '/behaviors.tsv'),
            sep='\t',
            names=['imp', 'uid', 'time', 'history', 'predict'],
            usecols=['imp', 'uid', 'predict']
        )

    def read_fuxi_novel_inter_data(self, mode):
        df = self.read_inter_data(mode)
        data = dict(
            imp=[],
            uid=[],
            nid=[],
            click=[]
        )
        for line in df.itertuples():
            predicts = line.predict.split(' ')
            data['imp'].extend([line.imp] * len(predicts))
            data['uid'].extend([line.uid] * len(predicts))
            for predict in predicts:
                if mode == 'test':
                    nid, click = predict, '0'
                else:
                    nid, click = predict.split('-')
                data['nid'].append(nid)
                data['click'].append(click)
        return pd.DataFrame(data)

    @staticmethod
    def build_news_ut(full_length=False):
        txt_tok = BertTok(name='english', vocab_dir='bert-base-uncased')

        return UniTok().add_index_col(
            name='nid'
        ).add_col(Column(
            name='cat',
            tokenizer=EntTok(name='cat').as_sing()
        )).add_col(Column(
            name='subcat',
            tokenizer=EntTok(name='subcat').as_sing(),
        )).add_col(Column(
            name='title',
            tokenizer=txt_tok.as_list(max_length=0 if full_length else 25),
        )).add_col(Column(
            name='abs',
            tokenizer=txt_tok.as_list(max_length=0 if full_length else 120),
        ))

    @staticmethod
    def build_user_ut(nid: Vocab, max_history: int = 0):
        user_ut = UniTok()
        user_ut.add_col(Column(
            name='uid',
            tokenizer=IdTok(name='uid').as_sing(),
        )).add_col(Column(
            name='history',
            tokenizer=SplitTok(
                name='newsList',
                sep=' ',
                vocab=nid
            ).as_list(max_length=max_history, slice_post=True),
        ))
        return user_ut

    @staticmethod
    def build_inter_ut(nid: Vocab, uid: Vocab):
        inter_ut = UniTok()
        inter_ut.add_col(Column(
            name='imp',
            tokenizer=IdTok(name='imp').as_sing(),
        )).add_col(Column(
            name='uid',
            tokenizer=EntTok(
                name='user',
                vocab=uid
            ).as_sing(),
        )).add_col(Column(
            name='predict',
            tokenizer=PredictTok(
                name='predictList',
                sep=' ',
                vocab=nid
            ).as_list()
        ))
        return inter_ut

    @staticmethod
    def build_fuxi_novel_inter_ut(nid: Vocab, uid: Vocab):
        click_vocab = Vocab(name='click')
        click_vocab.append('0')
        click_vocab.append('1')
        click_vocab.deny_edit()

        inter_ut = UniTok()
        inter_ut.add_index_col(
            name='index'
        )
        inter_ut.add_col(Column(
            name='imp',
            tokenizer=EntTok(
                name='imp'
            ).as_sing(),
        )).add_col(Column(
            name='uid',
            tokenizer=EntTok(
                name='user',
                vocab=uid
            ).as_sing(),
        )).add_col(Column(
            name='nid',
            tokenizer=EntTok(
                name='news',
                vocab=nid
            ).as_sing(),
        )).add_col(Column(
            name='click',
            tokenizer=EntTok(
                name='click',
                vocab=click_vocab,
            ).as_sing()
        ))
        return inter_ut

    def combine_news_df(self, with_body_data, test=False):
        news_train_df = self.read_news_data('train')
        news_dev_df = self.read_news_data('dev')
        if test:
            news_test_df = self.read_news_data('test')
            news_df = pd.concat([news_train_df, news_dev_df, news_test_df])
        else:
            news_df = pd.concat([news_train_df, news_dev_df])

        print('before', len(news_df))
        news_df = news_df.drop_duplicates(['nid'])
        print('after', len(news_df))

        if with_body_data:
            bodies = self.read_body_data()
            nids = news_df['nid']
            ordered_bodies = []
            for nid in nids:
                ordered_bodies.append(bodies[nid] or '')
            news_df['body'] = ordered_bodies

        return news_df

    def combine_user_df(self):
        user_train_df = self.read_user_data('train')
        user_dev_df = self.read_user_data('dev')

        user_df = pd.concat([user_train_df, user_dev_df])
        print('before', len(user_df))
        print('ensure', len(user_df.drop_duplicates()))
        user_df = user_df.drop_duplicates(['uid'])
        print('after', len(user_df))
        return user_df

    def get_test_user_df(self):
        user_df = self.read_user_data('test')
        print('before', len(user_df))
        print('ensure', len(user_df.drop_duplicates()))
        user_df = user_df.drop_duplicates(['uid'])
        print('after', len(user_df))
        return user_df

    def analyse_news(self, with_body_data=False):
        news_ut = self.build_news_ut(with_body_data=with_body_data)
        news_df = self.combine_news_df(with_body_data=with_body_data)
        news_ut.id_col.tok.vocab.allow_edit()

        news_ut.read_file(news_df).analyse()

    def analyse_user(self):
        user_ut = self.build_user_ut(nid=Vocab(name='nid'))
        user_df = self.combine_user_df()
        user_ut.id_col.tok.vocab.allow_edit()

        user_ut.read_file(user_df).analyse()

    def build_rec_data(self, test=False, max_history=0):
        news_ut = self.build_news_ut(with_body_data=True)
        news_ut.id_col.tok.vocab.allow_edit()
        news_df = self.combine_news_df(with_body_data=True, test=test)

        user_ut = self.build_user_ut(
            nid=news_ut.id_col.tok.vocab,
            max_history=max_history,
        )
        user_ut.id_col.tok.vocab.allow_edit()
        user_df = self.combine_user_df()

        inter_ut = self.build_fuxi_novel_inter_ut(
            nid=news_ut.id_col.tok.vocab,
            uid=user_ut.id_col.tok.vocab,
        )
        inter_train_df = self.read_fuxi_novel_inter_data('train')
        inter_dev_df = self.read_fuxi_novel_inter_data('dev')

        news_ut.read_file(news_df).tokenize().store_data(self.news_store_dir)
        news_ut.id_col.tok.vocab.deny_edit()
        user_ut.read_file(user_df).tokenize().store_data(self.user_store_dir)
        user_ut.id_col.tok.vocab.deny_edit()
        inter_ut.read_file(inter_train_df).tokenize().store_data(self.train_store_dir)
        inter_ut.read_file(inter_dev_df).tokenize().store_data(self.dev_store_dir)

        if test:
            user_test_ut = self.build_user_ut(
                nid=news_ut.id_col.tok.vocab,
                # nid=news_vocab,
                max_history=max_history,
            )
            user_test_df = self.get_test_user_df()
            user_test_ut.read_file(user_test_df).tokenize().store_data(self.user_test_store_dir)
            user_test_ut.id_col.tok.vocab.deny_edit()

            inter_test_ut = self.build_fuxi_novel_inter_ut(
                nid=news_ut.id_col.tok.vocab,
                # nid=news_vocab,
                uid=user_test_ut.id_col.tok.vocab,
            )
            inter_test_df = self.read_fuxi_novel_inter_data('test')
            inter_test_ut.read_file(inter_test_df).tokenize().store_data(self.test_store_dir)

    def build_rec_data_from_large(self):
        news_depot = UniDep(self.news_store_dir.replace('small', 'large'))
        news_vocab = news_depot.vocab_depot.get_vocab(news_depot.id_col)  # type: Vocab
        news_vocab.deny_edit()

        user_depot = UniDep(self.user_store_dir.replace('small', 'large'))
        user_vocab = user_depot.vocab_depot.get_vocab(user_depot.id_col)  # type: Vocab
        user_vocab.deny_edit()

        inter_ut = self.build_fuxi_novel_inter_ut(
            nid=news_vocab,
            uid=user_vocab,
        )
        inter_train_df = self.read_fuxi_novel_inter_data('train')
        inter_dev_df = self.read_fuxi_novel_inter_data('dev')

        inter_ut.read_file(inter_train_df).tokenize().store_data(self.train_store_dir)
        inter_ut.read_file(inter_dev_df).tokenize().store_data(self.dev_store_dir)

    def build_fuxi_novel_data(self, test=False):
        news_ut = self.build_news_ut(with_body_data=True)
        news_ut.id_col.tok.vocab.allow_edit()
        news_df = self.combine_news_df(with_body_data=True, test=test)

        user_ut = self.build_user_ut(
            nid=news_ut.id_col.tok.vocab
        )
        user_ut.id_col.tok.vocab.allow_edit()
        user_df = self.combine_user_df()

        inter_ut = self.build_fuxi_novel_inter_ut(
            nid=news_ut.id_col.tok.vocab,
            uid=user_ut.id_col.tok.vocab,
        )
        inter_train_df = self.read_fuxi_novel_inter_data('train')
        inter_dev_df = self.read_fuxi_novel_inter_data('dev')

        news_ut.read_file(news_df).tokenize().store_data(self.news_store_dir)
        news_ut.id_col.tok.vocab.deny_edit()
        user_ut.read_file(user_df).tokenize().store_data(self.user_store_dir)
        user_ut.id_col.tok.vocab.deny_edit()
        inter_ut.read_file(inter_train_df).tokenize().store_data(self.train_store_dir)
        inter_ut.read_file(inter_dev_df).tokenize().store_data(self.dev_store_dir)

        if test:
            inter_test_df = self.read_fuxi_novel_inter_data('test')
            inter_ut.read_file(inter_test_df).tokenize().store_data(self.test_store_dir)

    def tokenize(self, with_body_data=False):
        news_ut = self.build_news_ut(with_body_data=with_body_data)
        news_ut.id_col.tok.vocab.allow_edit()
        news_df = self.combine_news_df(with_body_data=with_body_data)

        user_ut = self.build_user_ut(
            nid=news_ut.id_col.tok.vocab
        )
        user_ut.id_col.tok.vocab.allow_edit()
        user_df = self.combine_user_df()

        inter_ut = self.build_inter_ut(
            nid=news_ut.id_col.tok.vocab,
            uid=user_ut.id_col.tok.vocab,
        )
        inter_train_df = self.read_inter_data('train')
        inter_dev_df = self.read_inter_data('dev')

        news_ut.read_file(news_df).tokenize().store_data(self.news_store_dir)
        news_ut.id_col.tok.vocab.deny_edit()
        user_ut.read_file(user_df).tokenize().store_data(self.user_store_dir)
        user_ut.id_col.tok.vocab.deny_edit()
        inter_ut.read_file(inter_train_df).tokenize().store_data(self.train_store_dir)
        inter_ut.read_file(inter_dev_df).tokenize().store_data(self.dev_store_dir)

    def _to_fuxi_line(self, with_body_data, sample=None):
        # attrs = ['imp', 'uid', 'nid', 'title', 'abs', 'cat', 'subCat', 'history_nids', 'history_titles', 'click']
        attrs = ['imp', 'uid', 'nid', 'cat', 'subCat', 'history_nids', 'click']
        if with_body_data:
            attrs.append('body')

        if not sample:
            return '\t'.join(attrs) + '\n'

        s = []
        for attr in attrs:
            s.append(str(sample[attr]))
        return '\t'.join(s) + '\n'

    def rebuild_mind_large_data(self, mode):
        if mode == 'train':
            depot = UniDep(self.train_store_dir)
        elif mode == 'dev':
            depot = UniDep(self.dev_store_dir)
        else:
            depot = UniDep(self.test_store_dir)

        index = []
        for i in range(depot.sample_size):
            if depot[i]['click'] == 0:
                if random.random() > 0.2:
                    continue
            index.append(i)

        s = os.path.join(depot.store_dir, 'data.new.npy')
        data = dict()
        for k in depot.data:
            data[k] = []
        for k in depot.data:
            for i in index:
                data[k].append(depot.data[k][i])
        for k in depot.data:
            data[k] = np.array(data[k])
        np.save(s, data, allow_pickle=True)

    def build_large_fuxi_data(self):
        attr_format = lambda tokens: '^'.join(map(str, tokens))

        # user_depot = UniDep(self.user_store_dir)
        news_depot = UniDep(self.news_store_dir)
        # train_depot = UniDep(self.train_store_dir)
        # dev_depot = UniDep(self.dev_store_dir)
        test_depot = UniDep(self.test_store_dir)
        user_test_depot = UniDep(self.user_test_store_dir)

        # f_dev = open(os.path.join(self.store_dir, 'fuxi-dev.csv'), 'a+')
        # f_dev.write(self._to_fuxi_line(with_body_data=False))
        # for inter_depot, mode in zip([train_depot, dev_depot, test_depot], ['train', 'dev', 'test']):
        # for inter_depot, mode in zip([train_depot, dev_depot], ['train', 'dev']):
        for inter_depot, mode in zip([test_depot], ['test']):
            # if mode == 'dev':
            #     f = open(os.path.join(self.store_dir, 'fuxi-train.csv'), 'a+')
            # else:
            f = open(os.path.join(self.store_dir, 'fuxi-{}.csv'.format(mode)), 'a+')
            f.write(self._to_fuxi_line(with_body_data=False))

            # imp = None
            for sample in tqdm(inter_depot):
                # if mode == 'dev':
                #     if sample['imp'] != imp:
                #         if random.random() < 0.01:
                #             cf = f_dev
                #         else:
                #             cf = f
                #         imp = sample['imp']
                # else:
                cf = f

                # if mode == 'test':
                #     user_info = user_test_depot[sample['uid']]
                # else:
                    # if sample['click'] == 0:
                    #     if random.random() > 0.04:
                    #         continue
                user_info = user_test_depot[sample['uid']]
                sample.update(user_info)

                news_info = news_depot[sample['nid']]
                sample.update(news_info)

                sample['history_nids'] = attr_format(sample['history'])

                cf.write(self._to_fuxi_line(False, sample))
            f.close()
        # f_dev.close()

    def build_fuxi_data(self, with_body_data=False):
        attr_format = lambda tokens: '^'.join(map(str, tokens))

        user_depot = UniDep(self.user_store_dir)
        news_depot = UniDep(self.news_store_dir)
        train_depot = UniDep(self.train_store_dir)
        dev_depot = UniDep(self.dev_store_dir)

        for inter_depot, mode in zip([train_depot, dev_depot], ['train', 'dev']):
            f = open(os.path.join(self.store_dir, 'fuxi-{}.csv'.format(mode)), 'w+')
            f.write(self._to_fuxi_line(with_body_data))
            for sample in tqdm(inter_depot):
                user_info = user_depot[sample['uid']]
                sample.update(user_info)

                history_nids = []
                history_titles = []
                for nid in sample['history']:
                    history_nids.append(nid)
                    history_titles.extend(news_depot[nid]['title'])

                del sample['history']
                sample['history_nids'] = attr_format(history_nids)
                sample['history_titles'] = attr_format(history_titles)

                predicts = sample['predict']
                del sample['predict']

                for predict in predicts:
                    news_info = news_depot[predict[0]]
                    sample.update(news_info)
                    sample['click'] = predict[1]

                    sample['title'] = attr_format(sample['title'])
                    sample['abs'] = attr_format(sample['abs'])

                    if with_body_data:
                        sample['body'] = attr_format(sample['body'])
                    elif 'body' in sample:
                        del sample['body']

                    f.write(self._to_fuxi_line(with_body_data, sample))
            f.close()

    def get_fetched_news(self):
        body_path = os.path.join(self.store_dir, 'body.txt')
        if not os.path.exists(body_path):
            return []

        fetched_news = set()
        with open(body_path, 'r') as f:
            for news in f:
                fetched_news.add(json.loads(news)['nid'])
        return fetched_news

    def append_fetch_news(self, nid, body):
        body_path = os.path.join(self.store_dir, 'body.txt')
        with open(body_path, 'a') as f:
            f.write(json.dumps(dict(nid=nid, body=body), ensure_ascii=False) + '\n')

    @staticmethod
    def _fetch_body(url):
        for _ in range(3):
            try:
                r = requests.get(url)
                html = r.content.decode()
                r.close()
                soup = Souper(html, 'html.parser')
                body = soup.find(id='maincontent').text
                body = body.strip()
                paras = body.split('\n')
                body = ' '.join(filter(lambda p: len(p.split(' ')) > 5, paras))
                return ' '.join(body.split(' ')[:2000])
            except Exception:
                continue
        return None

    def fetch_news_body(self):
        fetched_news = self.get_fetched_news()

        news_train_df = pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'train/news.tsv'),
            sep='\t',
            names=['nid', 'cat', 'subCat', 'title', 'abs', 'url', 'titEnt', 'absEnt'],
            usecols=['nid', 'url'],
        )
        news_dev_df = pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'dev/news.tsv'),
            sep='\t',
            names=['nid', 'cat', 'subCat', 'title', 'abs', 'url', 'titEnt', 'absEnt'],
            usecols=['nid', 'url'],
        )
        news_test_df = pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'test/news.tsv'),
            sep='\t',
            names=['nid', 'cat', 'subCat', 'title', 'abs', 'url', 'titEnt', 'absEnt'],
            usecols=['nid', 'url'],
        )

        news_df = pd.concat([news_train_df, news_dev_df, news_test_df])
        news_df = news_df.drop_duplicates(['nid'])
        total_news = len(news_df)

        def get_news():
            for news in news_df.iloc:
                yield news

        gen = get_news()

        for _ in tqdm(range(total_news)):
            news = gen.__next__()
            if news.nid in fetched_news:
                continue
            body = self._fetch_body(news.url)
            self.append_fetch_news(news.nid, body)

        return news_df

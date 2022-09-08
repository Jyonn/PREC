from processor.mind.processor import MindProcessor


def analyse_mind_small_dataset():
    p = MindProcessor(
        data_dir='/data1/USER/Data/MIND/MINDsmall/',
        store_dir='data/MIND/MINDsmall'
    )

    p.analyse_user()


def build_mind_small_dataset():
    p = MindProcessor(
        data_dir='/data1/USER/Data/MIND/MINDsmall/',
        store_dir='data/MIND/MINDsmall'
    )

    p.tokenize()


def build_mind_small_dataset_with_body():
    p = MindProcessor(
        data_dir='/data1/USER/Data/MIND/MINDsmall/',
        store_dir='data/MIND/MINDsmall-body',
        body_path='/data1/USER/Data/MIND/body.txt',
    )

    p.tokenize(with_body_data=True)


def build_mind_small_fuxi_dataset_with_body():
    p = MindProcessor(
        data_dir='/data1/USER/Data/MIND/MINDsmall/',
        store_dir='data/MIND/MINDsmall-body',
        body_path='/data1/USER/Data/MIND/body.txt',
    )
    p.build_fuxi_data(with_body_data=True)


def build_mind_small_fuxi_novel_dataset():
    p = MindProcessor(
        data_dir='/data1/USER/Data/MIND/MINDsmall/',
        store_dir='data/MIND/MINDsmall-fuxi',
        body_path='/data1/USER/Data/MIND/body.txt',
    )
    p.build_fuxi_novel_data()


def build_mind_large_fuxi_novel_dataset():
    p = MindProcessor(
        data_dir='/data1/USER/Data/MIND/MINDlarge/',
        store_dir='data/MIND/MINDlarge-fuxi',
        body_path='/data1/USER/Data/MIND/body.txt',
    )

    p.build_fuxi_novel_data(test=True)


def build_mind_small_rec_dataset():
    p = MindProcessor(
        data_dir='/data1/USER/Data/MIND/MINDsmall/',
        store_dir='data/MIND/MINDsmall-rec',
        body_path='/data1/USER/Data/MIND/body.txt',
    )
    p.build_rec_data(max_history=50)


def build_mind_large_rec_dataset_with_max_history():
    p = MindProcessor(
        data_dir='/data1/USER/Data/MIND/MINDlarge/',
        store_dir='data/MIND/MINDlarge-rec',
        body_path='/data1/USER/Data/MIND/body.txt',
    )
    p.build_rec_data(max_history=50, test=True)


def build_mind_small_rec_dataset_from_large():
    p = MindProcessor(
        data_dir='/data1/USER/Data/MIND/MINDsmall/',
        store_dir='data/MIND/MINDsmall-rec',
        body_path='/data1/USER/Data/MIND/body.txt',
    )
    p.build_rec_data_from_large()


def build_mind_large_fuxi_data():
    p = MindProcessor(
        data_dir='/data1/USER/Data/MIND/MINDlarge/',
        store_dir='data/MIND/MINDlarge-rec',
        body_path='/data1/USER/Data/MIND/body.txt',
    )
    # p.rebuild_mind_large_data('train')
    # p.rebuild_mind_large_data('dev')
    # p.rebuild_mind_large_data('test')
    p.build_large_fuxi_data()


# build_mind_small_rec_dataset()
build_mind_large_fuxi_data()

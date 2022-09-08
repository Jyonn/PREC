from processor.adressa.processor import AdressaProcessor
from utils import random_seed

p = AdressaProcessor(
    data_dir='/data1/USER/Data/Adressa/',
    store_dir='data/Adressa/',
)

random_seed.seeding(2021)

# p.parse_data()
# p.append_news_content()
# p.tokenize_news()


# # 4Week
# p.tokenize_user(
#     history_days=(0, 20),
#     user_path='4Week/user',
#     history_length=50,
# )
# p.tokenize_inter(
#     history_days=(0, 20),
#     train_days=(20, 24),
#     dev_days=(24, 28),
#     dev_test_ratio=0.2,
#     user_path='4Week/user',
#     inter_path='4Week/neg4',
#     neg_ratio=4,
# )
#
# p.build_fuxi_dataset(
#     user_path='4Week/user',
#     inter_path='4Week/neg4'
# )
#
# p.build_gnud_dataset(
#     user_path='4Week/user',
#     inter_path='4Week/neg4',
#     store_path='4Week/USER-4week',
# )


# p.tokenize_user(
#     history_days=(0, 5),
#     user_path='1Week/user',
#     history_length=10,
# )

#
# p.tokenize_user(
#     history_days=(0, 50),
#     user_path='10Week/user',
#     history_length=50,
# )
# p.tokenize_inter(
#     history_days=(0, 50),
#     train_days=(50, 60),
#     dev_days=(60, 70),
#     dev_test_ratio=0.2,
#     user_path='10Week/user',
#     inter_path='10Week/neg4',
#     neg_ratio=4,
# )
#

# p.tokenize_user(
#     history_days=(0, 6),
#     user_path='1Week/user-test',
#     history_length=10,
# )
#
# p.build_fuxi_dataset(
#     user_path='1Week/user',
#     user_test_path='1Week/user-test',
#     inter_path='1Week/neg4',
#     modes=['dev', 'test'],
# )
#
# p.build_gnud_dataset(
#     user_path='10Week/user',
#     inter_path='10Week/neg4',
#     store_path='10Week/USER-10week',
# )


# 4Week
p.tokenize_user(
    history_days=(0, 24),
    user_path='4Week-v2/user',
    history_length=50,
)

p.tokenize_inter(
    history_days=(0, 24),
    train_days=(24, 26),
    dev_days=(26, 28),
    dev_test_ratio=0.2,
    user_path='4Week-v2/user',
    inter_path='4Week-v2/neg4',
    neg_ratio=4,
)

p.build_fuxi_dataset(
    user_path='4Week-v2/user',
    inter_path='4Week-v2/neg4',
)

p.build_gnud_dataset(
    user_path='4Week-v2/user',
    inter_path='4Week-v2/neg4',
    store_path='4Week-v2/USER-4week-v2',
)

# import torch
# from torch.utils.data.sampler import Sampler
#
# from osdata.roi_data_layer.roibatchLoader import roibatchLoader
# from osdata.roi_data_layer.roidb import combined_roidb
#
#
# class sampler(Sampler):
#     def __init__(self, train_size, batch_size):
#         self.num_data = train_size
#         self.num_per_batch = int(train_size / batch_size)
#         self.batch_size = batch_size
#         self.range = torch.arange(0, batch_size).view(1, batch_size).long()
#         self.leftover_flag = False
#         if train_size % batch_size:
#             self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
#             self.leftover_flag = True
#
#     def __iter__(self):
#         rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
#         self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
#
#         self.rand_num_view = self.rand_num.view(-1)
#
#         if self.leftover_flag:
#             self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)
#
#         return iter(self.rand_num_view)
#
#     def __len__(self):
#         return self.num_data
#
#
# # %%
# BATCH_SIZE = 4
# NUM_WORKERS = 2
#
# SEEN = 1
# IMDB_NAME = "coco_2017_train"
# IMDBVAL_NAME = "coco_2017_minival"
#
# # %% create dataloader
#
# imdb, roidb, ratio_list, ratio_index, query = combined_roidb(IMDB_NAME, True, seen=SEEN)
# train_size = len(roidb)
# print('{:d} roidb entries'.format(len(roidb)))
# sampler_batch = sampler(train_size, BATCH_SIZE)
# dataset = roibatchLoader(roidb, ratio_list, ratio_index, query, BATCH_SIZE, imdb.num_classes, training=True)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
#                                          sampler=sampler_batch, num_workers=NUM_WORKERS)

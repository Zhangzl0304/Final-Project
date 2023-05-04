from os import listdir
from os.path import join
import pandas as pd
import torch.utils.data as data

# from utils import is_image_file


class DatasetFromFolder(data.Dataset):
    def __init__(self, mode):
        super(DatasetFromFolder, self).__init__()

        self.path = r"C:\University of Electronic Science and Technology of China\senior\final project\dataset\OUISIR dataset\AutomaticExtractionData_IMUZCenter"
        self.filenames = [x for x in listdir(self.path)]
        self.mode = mode

    def __getitem__(self, index):
        # print(join(self.path, self.filenames[index]))
        a0 = pd.read_csv(join(self.path, self.filenames[index]), skiprows=2)
        a0 = a0.iloc[:250, :]
        a0 = a0.to_numpy()
        if self.mode == 'NN':
          a0 = a0.flatten('f')
        else:
          pass
        return a0


    #
    # def __len__(self):
    #     return len(self.rgb_image_filenames)

a = DatasetFromFolder('NN')
names = a.filenames
ID_list = []
for i in range(len(names)):
    ID0 = names[i]
    ID = ID0[5:11]
    ID_list.append(ID)

ID_set = set(ID_list)
ID_list_unq = list(ID_set)
ID_list_unq.sort()


label_list = pd.read_csv('C:\\University of Electronic Science and Technology of China\\senior\\final project\\dataset\\OUISIR dataset\\IDGenderAgelist.csv')

label = label_list.iloc[:, 0]
label = list(label)
label_align = []
for items in label:
    items = str(items)
    items = items.zfill(6)
    label_align.append(items)


label_list1 = pd.read_csv('C:\\University of Electronic Science and Technology of China\\senior\\final project\\dataset\\OUISIR dataset\\IDGenderAgelist.csv', index_col=0)

label_avail = [v for v in label_align if v in ID_list_unq]
label_avail_int = []
for i in label_avail:
    i2 = int(i)
    label_avail_int.append(i2)

label_avail01 = label_list1.loc[label_avail_int]

# label extract ========================================================
data_label = label_avail01['Gender(0:Female;1:Male)'].to_numpy()
data_label_2 = []
for i in range(len(data_label)):
    for j in range(2):
        data_label_2.append(data_label[i])
# print(data_label, len(data_label))
# print(data_label_2, len(data_label_2))
# label ================================================================

# data_ID = label_avail01.index.to_numpy()
#
# ID_list_int = []
# for i in ID_list:
#     i2 = int(i)
#     ID_list_int.append(i2)
#
# for j in range(1488):
#     if data_ID[int(j/2)] != ID_list_int[j]:
#         print('noo',data_ID[j],ID_list_int[int(j/2)],j,int(j/2))



import numpy as np
import struct

# convert bytefile to floatfile & reshape
def CRS_data(bytefile):
    bytefile.seek(0, 2)
    num_bytes2 = bytefile.tell()
    # print(num_bytes2)
    bytefile.seek(0)
    i = 0
    BS_vector = []

    while i < num_bytes2:
        data = bytefile.read(4)
        i += 4
        unpacked = struct.unpack("<f", data)
        BS_vector.append(unpacked)

    # reshpae 3-dimensional array to 2-dimensional array
    F_S = np.array(BS_vector)
    Reshape_F_S = F_S.reshape(len(BS_vector))

    return Reshape_F_S

# indexing & labeling
def IL_data(filename, num_label, period=1999):
    data = []
    cur_index = 1
    total_index = filename.shape[0]

    while cur_index < total_index - period:
        unlabeled_data = filename[cur_index:cur_index + period]
        labeled_data = np.append(unlabeled_data, num_label)
        data.append(labeled_data)
        cur_index = cur_index + period + 2

    array_data = np.array(data)
    return array_data

TR1 = open('rf_433_01', 'rb')
TR2 = open('rf_433_02', 'rb')
TR3 = open('rf_433_03', 'rb')
TR4 = open('rf_433_04', 'rb')
TR5 = open('rf_433_05', 'rb')
TR6 = open('rf_433_06', 'rb')
TR7 = open('rf_433_07', 'rb')
TR8 = open('rf_433_08', 'rb')
TR9 = open('rf_433_09', 'rb')

converted_data_1 = CRS_data(TR1)
data_001 = IL_data(converted_data_1, 0)
# print(data_001.shape)
converted_data_2 = CRS_data(TR2)
data_002 = IL_data(converted_data_2, 1)
# print(data_002.shape)
converted_data_3 = CRS_data(TR3)
data_003 = IL_data(converted_data_3, 2)
# print(data_003.shape)
converted_data_4 = CRS_data(TR4)
data_004 = IL_data(converted_data_4, 3)
# print(data_004.shape)
converted_data_5 = CRS_data(TR5)
data_005 = IL_data(converted_data_5, 4)
# print(data_005.shape)
converted_data_6 = CRS_data(TR6)
data_006 = IL_data(converted_data_6, 5)
# print(data_006.shape)
converted_data_7 = CRS_data(TR7)
data_007 = IL_data(converted_data_7, 6)
# print(data_007.shape)
converted_data_8 = CRS_data(TR8)
data_008 = IL_data(converted_data_8, 7)
# print(data_008.shape)
converted_data_9 = CRS_data(TR9)
data_009 = IL_data(converted_data_9, 8)
# print(data_009.shape)
'''
1번 29739개
2번 22255개
3번 22400개
4번 23667개
5번 24114개
6번 21688개
7번 22498개
8번 22287개
9번 21564개 
'''

combined_data = np.concatenate((data_001[:20000,:],
                                data_002[:20000,:],
                                data_003[:20000,:],
                                data_004[:20000,:],
                                data_005[:20000,:],
                                data_006[:20000,:],
                                data_007[:20000,:],
                                data_008[:20000,:],
                                data_009[:20000,:]), axis=0)
print(combined_data.shape)

num_shuffle = 100
for i in range(num_shuffle):
    np.random.shuffle(combined_data)

np.save('processed_data', combined_data)

print("Data is shuffled {}times, processed and saved!".format(num_shuffle))
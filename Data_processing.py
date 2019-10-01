import numpy as np
import struct

TR1 = open('rf_433_01', 'rb')
TR2 = open('rf_433_02', 'rb')
TR3 = open('rf_433_03', 'rb')
TR4 = open('rf_433_04', 'rb')
TR5 = open('rf_433_05', 'rb')
TR6 = open('rf_433_06', 'rb')
TR7 = open('rf_433_07', 'rb')
TR8 = open('rf_433_08', 'rb')
TR9 = open('rf_433_09', 'rb')

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


converted_data_1 = CRS_data(TR1)
data_001 = IL_data(converted_data_1, 1)

converted_data_2 = CRS_data(TR2)
data_002 = IL_data(converted_data_2, 2)

converted_data_3 = CRS_data(TR3)
data_003 = IL_data(converted_data_3, 3)

converted_data_4 = CRS_data(TR4)
data_004 = IL_data(converted_data_4, 4)

converted_data_5 = CRS_data(TR5)
data_005 = IL_data(converted_data_5, 5)

converted_data_6 = CRS_data(TR6)
data_006 = IL_data(converted_data_6, 6)

converted_data_7 = CRS_data(TR7)
data_007 = IL_data(converted_data_7, 7)

converted_data_8 = CRS_data(TR8)
data_008 = IL_data(converted_data_8, 8)

converted_data_9 = CRS_data(TR9)
data_009 = IL_data(converted_data_9, 9)

# print(len(converted_data_0))
# print(data_000.shape)
# print(data_000[23, 1990:2001])
#
# print(len(converted_data_1))
# print(data_001.shape)
# print(data_001[23, 1990:2001])
#
# print(len(converted_data_2))
# print(data_002.shape)
# print(data_002[23, 1990:2001])

combined_data = np.concatenate((data_001, data_002, data_003, data_004, data_005, data_006, data_007, data_008, data_009), axis=0)
print(combined_data.shape)

for i in range(5):
    np.random.shuffle(combined_data)

np.save('processed_data', combined_data)
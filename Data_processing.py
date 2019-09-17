import numpy as np
import struct
import matplotlib.pyplot as plt
import math

TR0 = open('rf_433_test_000', 'rb')
TR1 = open('rf_433_test_001', 'rb')
TR2 = open('rf_433_test_002', 'rb')

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

converted_data_0 = CRS_data(TR0)
data_000 = IL_data(converted_data_0, 0)

converted_data_1 = CRS_data(TR1)
data_001 = IL_data(converted_data_1, 1)

converted_data_2 = CRS_data(TR2)
data_002 = IL_data(converted_data_2, 2)

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

combined_data = np.concatenate((data_000, data_001, data_002), axis=0)
print(combined_data.shape)

np.random.shuffle(combined_data)
np.random.shuffle(combined_data)
np.random.shuffle(combined_data)
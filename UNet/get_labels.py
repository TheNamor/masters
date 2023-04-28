import os

CRC_PATH = 'C:/Users/TheNa/Desktop/CRC_Dataset'
crc_ids = os.listdir(CRC_PATH)

labels = [0 if "Normal" in name else (1 if "Low" in name else 2) for name in [crc_ids[i] for i in range(len(crc_ids)) if i!=106]]

print(sum(labels), len(labels)-sum(labels), len(labels))

print(labels)
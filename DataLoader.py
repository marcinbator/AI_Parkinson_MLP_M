import hickle as hkl
import numpy as np
from matplotlib import pyplot as plt


class DataLoader:
    __filename: str = None
    __targetData = None
    __sortedData = None
    __sortedTargetData = None

    def __init__(self, filename):
        self.__filename = filename

    def loadAndSaveData(self, showPlots):
        dataAsText = np.loadtxt(self.__filename, delimiter=',', dtype=str)

        dataArray = dataAsText.astype(float).T
        dataArray = np.delete(dataArray, 16, 0)
        data = self.__normalizeData(dataArray)

        targetData = dataAsText[:, -7].astype(float)

        self.__targetData = targetData.reshape(1, targetData.shape[0])
        self.__sortedData, self.__sortedTargetData = self.__sortData(data, self.__targetData)

        self.__saveDataToHkl()

        if showPlots:
            self.__plotTargetData(self.__targetData)
            self.__plotTargetData(self.__sortedTargetData)

    def __saveDataToHkl(self):
        output_filename = self.__getOutputFileName()
        hkl.dump([self.__sortedData, self.__sortedTargetData], output_filename)

    def __getOutputFileName(self):
        return "".join(self.__filename.split(".")[0]) + '.hkl'

    @staticmethod
    def __sortData(data, target):
        sort_argument = np.argsort(target[0])
        data_sorted = data[:, sort_argument]
        target_sorted = target[:, sort_argument]
        return data_sorted, target_sorted

    @staticmethod
    def __normalizeData(data):
        x_min = data.min(axis=1)
        x_max = data.max(axis=1)
        x_norm_max = 1
        x_norm_min = -1
        x_norm = np.zeros(data.shape)
        for i in range(data.shape[0]):
            x_norm[i, :] = (x_norm_max - x_norm_min) / (x_max[i] - x_min[i]) * \
                           (data[i, :] - x_min[i]) + x_norm_min
        return data

    @staticmethod
    def __plotTargetData(targetData):
        plt.plot(targetData[0])
        plt.title("Loaded target data")
        plt.xlabel("Instance")
        plt.ylabel("Value")
        plt.show()

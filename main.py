import hickle as hkl

from DataLoader import DataLoader

dataProcessor = DataLoader("parkinsons.data")
data = None
target = None


def loadData():
    global data, target

    dataProcessor.loadAndSaveData(True)
    data, target = hkl.load('parkinsons.hkl')


loadData()
print("Data loaded.")

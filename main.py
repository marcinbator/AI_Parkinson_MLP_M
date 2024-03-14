import hickle as hkl

from DataLoader import DataLoader

dataProcessor = DataLoader("data/parkinsons.data")
data = None
target = None


def loadData():
    global data, target

    dataProcessor.loadAndSaveData(True)
    data, target = hkl.load('data/parkinsons.hkl')


loadData()
print("Data loaded.")

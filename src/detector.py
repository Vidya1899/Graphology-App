import os 

def runInference(latestFile):
    return {"fileList":os.listdir("uploads/"), "latestFile":latestFile}

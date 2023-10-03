
import time

def getTime(currentTime, labelName, printTime=True, spaceSize=50):
    '''Print the elapsed time since currentTime. Return the new current time.'''
    if printTime:
        print(labelName, ' ' * (spaceSize - len(labelName)), ': ', round((time.perf_counter() - currentTime) * 1000, 2), 'milliseconds')
    return time.perf_counter()
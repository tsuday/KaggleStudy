import numpy as np
import csv

file_train = 'train.csv'
file_test = 'test.csv'
file_result = 'result_np.csv'

## learn
f = open(file_train, 'r')
reader = csv.reader(f)

count = np.zeros(10)
pixels = np.zeros([10, 784])

next(reader) # read header

# calculate average pixel values
for r in reader:
    cols = list(map(int, r[:]))
    number = cols[0]
    count[number] += 1
    pixels[number] += cols[1:]

for i in range(10):
        pixels[i] = pixels[i] / count[i]

## predict
fp = open(file_test, 'r')
fresult = open(file_result, 'w')
readerp = csv.reader(fp)

next(readerp) # read header

fresult.write('ImageId,Label\n')

row=1
for r in readerp:
    test_pics = np.array(list(map(int, r)))
    pic = np.zeros([10, 784])
    for i in range(10):
        pic[i] = test_pics

    diffMin = 255*255*784+1
    diffMinIndex = -1
    # to avoid overflow, difference of pixel values are divided by 255
    arDiff = np.sum(((pixels-pic)/255.0)*((pixels-pic)/255.0)*((pixels-pic)/255.0)*((pixels-pic)/255.0), axis=1)
    for nm in range(len(pixels)): # 10
        diff = arDiff[nm]
        if diffMin > diff:
            diffMin = diff
            diffMinIndex = nm

    fresult.write(str(row)+","+str(diffMinIndex)+"\n")
    row += 1

fresult.close()
fp.close()
f.close()

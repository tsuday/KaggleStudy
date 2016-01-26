import csv

file_train = 'train.csv'
file_test = 'test.csv'
file_result = 'result.csv'

## learn
f = open(file_train, 'r')
reader = csv.reader(f)

count = [0] * 10
pixels = {}
for num in range(10):
    pixels[num] = [0] * 784

next(reader) # read header

# calculate average pixel values
for r in reader:
    cols = list(map(int, r[:]))
    number = cols[0]
    count[number] += 1
    for i in range(len(pixels[number])):
        pixels[number][i] += cols[1:][i]

for i in range(10):
    for j in range(784):
        pixels[i][j] = pixels[i][j] / count[i]

## predict
fp = open(file_test, 'r')
fresult = open(file_result, 'w')
readerp = csv.reader(fp)

next(readerp) # read header

fresult.write('ImageId,Label\n')

row=1
for r in readerp:
    test_pics = list(map(int, r))
    diffMin = 255*255*784+1
    diffMinIndex = -1
    for nm in range(len(pixels)): # 10
        diff = 0.0
        for pc in range(784):
            diff += (pixels[nm][pc]-test_pics[pc])*(pixels[nm][pc]-test_pics[pc])

        if diffMin > diff:
            diffMin = diff
            diffMinIndex = nm

    fresult.write(str(row)+","+str(diffMinIndex)+"\n")
    row += 1


fresult.close()
fp.close()
f.close()

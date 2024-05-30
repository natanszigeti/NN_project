file = open("data.txt", "r")
line = file.readline()
while line != "":
    halfLine = line[0:int(len(line)/2)]
    newLine = ""
    for i in range(len(line)-1):
        if i % 2 == 0:
            newLine += halfLine[int(i/2)]
        else:
            newLine += "0"
    print(newLine)
    line = file.readline()

file.close()
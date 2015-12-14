import sklearn
import sys
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

def isFloat(value):
  try:
    float(value)
    return True
  except:
    return False

def isInt(value):
  try:
    int(value)
    return True
  except:
    return False

def isNum(value):
    return isInt(value) or isFloat(value)


labels = []
samples = []
t_correct_labels = []
t_samples = []
types = ["ddm","er","geo","pam"]

train_test = "train"
next = "label"

for type in types:
    f = open(type+"_normal/output.txt","r")
    for line in f:
        if train_test == "train":
            if next == "label":
                next = "sample"
                if len(line) <= 5:
                    labels.append(line.rstrip())
                else:
                    print("label,invalid input data:",line)
                    sys.exit(0)
            elif next == "sample":
                next = "label"
                train_test = "test"
                if len(line) <= 5:
                    print("sample,invalid input data",line)
                    sys.exit(0)
                else:
                    sample = []
                    split_line = line.rstrip().replace(' ', '').replace('[', '').replace(']', '').replace("'ND'", "0").split(',')
                    for elmt in split_line:
                        if "j" in elmt:
                            sample.append(complex(elmt).real)
                        elif isNum(elmt):
                            sample.append(float(elmt))
                        else:
                            print(elmt)
                            sys.exit(0)
                    # samples.append(sample[:])
                    # related to nodes
                    samples.append([sample[0],sample[3],sample[4],sample[18],sample[19]])
                    # related to shape
                    # samples.append([sample[12],sample[20]/sample[0],sample[21]/sample[0],sample[22]])
                    # samples.append([sample[0],sample[12],sample[20],sample[21],sample[22]])
            else:
                print("error,invalid input data",line)
                sys.exit(0)

        elif train_test == "test":
            if next == "label":
                next = "sample"
                if len(line) <= 5:
                    t_correct_labels.append(line.rstrip())
                else:
                    print("label,invalid input data",line)
                    sys.exit(0)
            elif next == "sample":
                next = "label"
                train_test = "train"
                if len(line) <= 5:
                    print("sample,invalid input data",line)
                    sys.exit(0)
                else:
                    sample = []
                    split_line = line.rstrip().replace(' ', '').replace('[', '').replace(']', '').replace("'ND'", "0").split(',')
                    for elmt in split_line:
                        if "j" in elmt:
                            sample.append(complex(elmt).real)
                        elif isNum(elmt):
                            sample.append(float(elmt))
                        else:
                            print(elmt)
                            sys.exit(0)
                    #t_samples.append(sample[:])
                    # related to nodes
                    t_samples.append([sample[0],sample[3],sample[4],sample[18],sample[19]])
                    # related to shape
                    # t_samples.append([sample[12],sample[20]/sample[0],sample[21]/sample[0],sample[22]])
                    # t_samples.append([sample[0],sample[12],sample[20],sample[21],sample[22]])
            else:
                print("error,invalid input data",line)
                sys.exit(0)

    f = open(type+"_rewire/output.txt","r")
    for line in f:
        if train_test == "train":
            if next == "label":
                next = "sample"
                if len(line) <= 5:
                    labels.append(line.rstrip())
                else:
                    print("label,invalid input data:",line)
                    sys.exit(0)
            elif next == "sample":
                next = "label"
                train_test = "test"
                if len(line) <= 5:
                    print("sample,invalid input data",line)
                    sys.exit(0)
                else:
                    sample = []
                    split_line = line.rstrip().replace(' ', '').replace('[', '').replace(']', '').replace("'ND'", "0").split(',')
                    for elmt in split_line:
                        if "j" in elmt:
                            sample.append(complex(elmt).real)
                        elif isNum(elmt):
                            sample.append(float(elmt))
                        else:
                            print(elmt)
                            sys.exit(0)
                    # samples.append(sample[:])
                    # related to nodes
                    samples.append([sample[0],sample[3],sample[4],sample[18],sample[19]])
                    # related to shape
                    # samples.append([sample[12],sample[20]/sample[0],sample[21]/sample[0],sample[22]])
                    # samples.append([sample[0],sample[12],sample[20],sample[21],sample[22]])
            else:
                print("error,invalid input data",line)
                sys.exit(0)

        elif train_test == "test":
            if next == "label":
                next = "sample"
                if len(line) <= 5:
                    t_correct_labels.append(line.rstrip())
                else:
                    print("label,invalid input data",line)
                    sys.exit(0)
            elif next == "sample":
                next = "label"
                train_test = "train"
                if len(line) <= 5:
                    print("sample,invalid input data",line)
                    sys.exit(0)
                else:
                    sample = []
                    split_line = line.rstrip().replace(' ', '').replace('[', '').replace(']', '').replace("'ND'", "0").split(',')
                    for elmt in split_line:
                        if "j" in elmt:
                            sample.append(complex(elmt).real)
                        elif isNum(elmt):
                            sample.append(float(elmt))
                        else:
                            print(elmt)
                            sys.exit(0)
                    # t_samples.append(sample[:])
                    # related to nodes
                    t_samples.append([sample[0],sample[3],sample[4],sample[18],sample[19]])
                    # related to shape
                    # t_samples.append([sample[12],sample[20]/sample[0],sample[21]/sample[0],sample[22]])
                    # t_samples.append([sample[0],sample[12],sample[20],sample[21],sample[22]])
            else:
                print("error,invalid input data",line)
                sys.exit(0)

maxs = [-9999 for i in range(len(samples[0]))]
mins = [9999 for i in range(len(samples[0]))]
for s in samples:
    for i in range(len(s)):
        if i == 23:
            s[i] = s[i] + 1
        if s[i] < 0:
            print("element should not be negtive")
            sys.exit(0)
        if s[i] > maxs[i]:
            maxs[i] = s[i]
        if s[i] < mins[i]:
            mins[i] = s[i]
for s in samples:
    for i in range(len(s)):
        if maxs[i]+mins[i] != 0:
            s[i] = s[i]/(maxs[i]+mins[i])


maxs = [-9999 for i in range(len(samples[0]))]
mins = [9999 for i in range(len(samples[0]))]
for s in t_samples:
    for i in range(len(s)):
        if i == 23:
            s[i] = s[i] + 1
        if s[i] < 0:
            print("element should not be negtive")
            sys.exit(0)
        if s[i] > maxs[i]:
            maxs[i] = s[i]
        if s[i] < mins[i]:
            mins[i] = s[i]
for s in t_samples:
    for i in range(len(s)):
        if maxs[i]+mins[i] != 0:
            s[i] = s[i]/(maxs[i]+mins[i])

for l in range(len(samples)):
    print(labels[l],samples[l])

if len(labels) != len(samples):
    print(labels)
    print("invalid data, labels",len(labels),"samples:",len(samples))
    sys.exit(0)

print("training...")
gnb.fit(samples, labels)

if len(t_correct_labels) != len(t_samples):
    print("invalid data, labels",len(t_correct_labels),"samples:",len(t_samples))
    sys.exit(0)

print("predicting...")
ret_labes = gnb.predict(t_samples)

suc = 0
fail = 0
for i in range(len(t_correct_labels)):
    if t_correct_labels[i] == ret_labes[i]:
        suc += 1
    else:
        fail += 1
print("succed:",suc,"fail",fail,"rate:",suc/(fail+suc))



# next = "label"
# f = open("test/data.txt","r")
# for line in f:
#     if next == "label":
#         next = "sample"
#         if len(line) <= 5:
#             t_correct_labels.append(line.rstrip())
#         else:
#             print("invalid input data")
#             sys.exit(0)
#     elif next == "sample":
#         next = "label"
#         if len(line) <= 5:
#             print("invalid input data")
#             sys.exit(0)
#         else:
#             sample = []
#             split_line = line.rstrip().replace(' ', '').replace('[', '').replace(']', '').replace("'ND'", "0").split(',')
#             for elmt in split_line:
#                 if "j" in elmt:
#                     sample.append(complex(elmt).real)
#                 elif isNum(elmt):
#                     sample.append(float(elmt))
#                 else:
#                     print(elmt)
#                     sys.exit(0)
#             t_samples.append(sample[:-1])
#     else:
#         print("invalid input data")
#         sys.exit(0)




# num_test = 50
# traindata = []
# label = []


# rawdata = fp1.read().split('\n')

# for idx , data in enumerate(rawdata):
#     if idx % 2 == 1:
#         #print(idx)
#         tmp = data.replace("[","")
#         tmp = tmp.replace("]","")
#         #try:
#         tmp = map(float,(tmp.split(',')))
#         traindata.append(tmp)
#         #except Exception:
#         #    pass
#     else:
#         label.append(data)

# label = list(map(int,label))

# #for idx , data in enumerate(traindata):
# #    print(traindata[idx][0],traindata[idx][1],10000*traindata[idx][27]/traindata[idx][1],traindata[idx][27],1000*traindata[idx][12])

# idx2 = label.index(2)
# end2 = len(label)
# #print(idx2,end2)

# training = traindata[1:(idx2-num_test)]
# trainlabel = label[1:(idx2-num_test)]
# training.extend(traindata[idx2:end2-num_test-1])
# trainlabel.extend(label[idx2:end2-num_test-1])

# testing = traindata[(idx2-num_test):(idx2-1)]
# testlabel = label[(idx2-num_test):(idx2-1)]
# testing.extend(traindata[(end2-num_test):-1])
# testlabel.extend(label[(end2-num_test):-1])

# print(len(trainlabel))

# X = np.array(training)
# Y = np.array(trainlabel)

# print(X)

# clf = svm.SVC(kernel = 'linear',C = 1.0)
# clf.fit(X,Y)

# ret = clf.predict(testing)
# print(ret)
# print(testlabel)


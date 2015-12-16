import sklearn
import sys
import numpy as np
import math
from sklearn import svm

clf = svm.SVC(kernel = 'linear',C = 1.0)
#clf = svm.SVC(kernel = 'rbf',C = 1.0)

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

# assume there are only 4 families.
def precision_recall(truth, fact):
    no = 4
    precision = 0.0
    recall = 0.0
    # left_right : left is truth, right is fact
    a_a = 0
    a_b = 0
    a_c = 0
    a_d = 0
    b_a = 0
    b_b = 0
    b_c = 0
    b_d = 0
    c_a = 0
    c_b = 0
    c_c = 0
    c_d = 0
    d_a = 0
    d_b = 0
    d_c = 0
    d_d = 0
    truth = list(map(int, truth))
    fact = list(map(int, fact))
    for i in range(len(truth)):
        if truth[i] == 1 and fact[i] == 1:
            a_a += 1
        if truth[i] == 1 and fact[i] == 2:
            a_b += 1
        if truth[i] == 1 and fact[i] == 3:
            a_c += 1
        if truth[i] == 1 and fact[i] == 4:
            a_d += 1
        if truth[i] == 2 and fact[i] == 1:
            b_a += 1
        if truth[i] == 2 and fact[i] == 2:
            b_b += 1
        if truth[i] == 2 and fact[i] == 3:
            b_c += 1
        if truth[i] == 2 and fact[i] == 4:
            b_d += 1
        if truth[i] == 3 and fact[i] == 1:
            c_a += 1
        if truth[i] == 3 and fact[i] == 2:
            c_b += 1
        if truth[i] == 3 and fact[i] == 3:
            c_c += 1
        if truth[i] == 3 and fact[i] == 4:
            c_d += 1
        if truth[i] == 4 and fact[i] == 1:
            d_a += 1
        if truth[i] == 4 and fact[i] == 2:
            d_b += 1
        if truth[i] == 4 and fact[i] == 3:
            d_c += 1
        if truth[i] == 4 and fact[i] == 4:
            d_d += 1
    tp = a_a + b_b + c_c + d_d
    fp = b_a + c_a + d_a + c_b + d_b + d_c
    fn = a_b + a_c + a_d + b_c + b_d + c_d
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

labels = []
samples = []
t_correct_labels = []
t_samples = []
types = ["pam","er","geo","ddm"]
#types = ["ddm"]

train_test = "train"
next = "label"
#shape
#cols = [13, 22, 23]
#node and edge
#cols = [19,20,4,5,26,34,35,17]
#path
cols =[6,26,27]
#eigen 
#cols = [7,14,15]
#combination 
#cols = [13, 22, 23,19,20,4,5,26,34,35,17,6,26,27,7,14,15]

#cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]

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
                    # samples.append([sample[0],sample[3],sample[4],sample[18],sample[19]])
                    # related to shape
                    # samples.append([sample[12],sample[20]/sample[0],sample[21]/sample[0],sample[22]])
                    # samples.append([sample[0],sample[12],sample[20],sample[21],sample[22]])
                    f = []
                    for col in cols:
                        f.append(sample[col-1])
                    samples.append(f[:])
                    f = []
                    

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
                    # t_samples.append([sample[0],sample[3],sample[4],sample[18],sample[19]])
                    # related to shape
                    # t_samples.append([sample[12],sample[20]/sample[0],sample[21]/sample[0],sample[22]])
                    # t_samples.append([sample[0],sample[12],sample[20],sample[21],sample[22]])
                    f = []
                    for col in cols:
                        f.append(sample[col-1])
                    t_samples.append(f[:])
                    f = []
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
                    # samples.append([sample[0],sample[3],sample[4],sample[18],sample[19]])
                    # related to shape
                    # samples.append([sample[12],sample[20]/sample[0],sample[21]/sample[0],sample[22]])
                    # samples.append([sample[0],sample[12],sample[20],sample[21],sample[22]])
                    f = []
                    for col in cols:
                        f.append(sample[col-1])
                    samples.append(f[:])
                    f = []
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
                    # t_samples.append([sample[0],sample[3],sample[4],sample[18],sample[19]])
                    # related to shape
                    # t_samples.append([sample[12],sample[20]/sample[0],sample[21]/sample[0],sample[22]])
                    # t_samples.append([sample[0],sample[12],sample[20],sample[21],sample[22]])
                    f = []
                    for col in cols:
                        f.append(sample[col-1])
                    t_samples.append(f[:])
                    f = []
            else:
                print("error,invalid input data",line)
                sys.exit(0)
    ###
    #total_node = sample[0]+total_node;
    #print("ave node",total_node)
    #total_edge = sample[0]+total_edge;  
    #print("ave edge",total_edge)
    ###

maxs = [-9999 for i in range(len(samples[0]))]
mins = [9999 for i in range(len(samples[0]))]
########
means = [0 for i in range(len(samples[0]))]
varss = [0 for i in range(len(samples[0]))]



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

########        ####
        means[i] += s[i]  
for m in range(len(means)): 
    means[m] = means[m]/len(samples[0])
    ("mean",means[m])
for s in samples:
    for i in range(len(s)):
        varss[i] = (s[i]-means[i])*(s[i]-means[i])
        #print(varss[i])

for v in range(len(varss)): 
    varss[v] = math.sqrt(varss[v])
    print("var",varss[v])

for s in samples:
    for i in range(len(s)):
        if varss[i] != 0:
            s[i] = abs(s[i]-means[i])/varss[i]
######        ####
# for s in samples:
#     for i in range(len(s)):
#         if maxs[i]+mins[i] != 0:
#             s[i] = s[i]/(maxs[i]+mins[i])


maxs = [-9999 for i in range(len(samples[0]))]
mins = [9999 for i in range(len(samples[0]))]
########
means = [0 for i in range(len(samples[0]))]
varss = [0 for i in range(len(samples[0]))]
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

########        ####
        means[i] += s[i]  
for m in range(len(means)): 
    means[m] = means[m]/len(samples[0])

for s in t_samples:
    for i in range(len(s)):
        varss[i] = (s[i]-means[i])*(s[i]-means[i])

for v in range(len(varss)): 
    varss[v] = math.sqrt(varss[v])

for s in t_samples:
    for i in range(len(s)):
        if varss[i] != 0:
            s[i] = abs(s[i]-means[i])/varss[i]
######        ####
# for s in t_samples:
#     for i in range(len(s)):
#         if maxs[i]+mins[i] != 0:
#             s[i] = s[i]/(maxs[i]+mins[i])


for l in range(len(samples)):
    print(labels[l],samples[l])

if len(labels) != len(samples):
    print(labels)
    print("invalid data, labels",len(labels),"samples:",len(samples))
    sys.exit(0)

print("training...")
clf.fit(samples, labels)

if len(t_correct_labels) != len(t_samples):
    print("invalid data, labels",len(t_correct_labels),"samples:",len(t_samples))
    sys.exit(0)



print("predicting...")
ret_labes = clf.predict(t_samples)

##### AUPR ####
precision, recall = precision_recall(t_correct_labels, ret_labes)
print("precision: {0}, recall: {1}".format(precision, recall))

########

suc = 0
fail = 0
#####
PAM_fail = 0
ER_fail = 0
GRO_fail = 0
DDM_fail = 0
####
for i in range(len(t_correct_labels)):
    if t_correct_labels[i] == ret_labes[i]:
        suc += 1
    else:
        fail += 1
        #print(t_correct_labels[i],ret_labes[i])
    #######    
        # if t_correct_labels[i] == 1: 
        #     PAM_fail += 1
        # if t_correct_labels[i] == 2: 
        #     ER_fail += 1
        # if t_correct_labels[i] == 3: 
        #     GRO_fail += 1
        # if t_correct_labels[i] == 4: 
        #     DDM_fail += 1
    ######
print("succed:",suc,"fail",fail,"rate:",suc/(fail+suc))
# print("PAM_fail",PAM_fail)
# print("ER_fail",ER_fail)
# print("GRO_fail",GRO_fail)
# print("DDM_fail",DDM_fail)


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

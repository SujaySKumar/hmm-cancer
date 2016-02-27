import pickle
import scipy.cluster.vq as sp
import numpy as np
f= open("testing_data.txt","r")
l=f.readlines()
test=[]
output=[]
codebooks = pickle.load(open("all_codebooks.pkl","rb"))
num_correct=0
for line in l:
    line=line.replace("\n","")
    line=line.replace("null","0")

    test=map(float,line.split(",")[1:])
    output = line.split(",")[0]
    if True:
        #print test
        try:
            vq_data1=map(str,sp.vq(np.reshape(test,(len(test)/8,8)),codebooks[8])[0])
        except:
            pass
        try:
            vq_data2=map(str,sp.vq(np.reshape(test,(len(test)/57,57)),codebooks[57])[0])
        except:
            pass
        try:
            vq_data3=map(str,sp.vq(np.reshape(test,(len(test)/18,18)),codebooks[18])[0])
        except:
            pass
        try:
            vq_data4=map(str,sp.vq(np.reshape(test,(len(test)/18,18)),codebooks[18])[0])
        except:
            pass
            #print vq_data
        f1 = open("prostate_trained.pkl","rb")
        f2 = open("breast_trained.pkl","rb")
        f3 = open("lung_trained.pkl","rb")
        f4 = open("colorectal_trained.pkl","rb")

        hmm1=pickle.load(f1)
        hmm2=pickle.load(f2)
        hmm3=pickle.load(f3)
        hmm4=pickle.load(f4)
        x=[]
        import math
        try:
            x.append(math.exp(hmm1.backward_scaled(vq_data1)))
        except:
            x.append(0)
        try:
            x.append(math.exp(hmm2.backward_scaled(vq_data2)))
        except:
            x.append(0)
        try:
            x.append(math.exp(hmm3.backward_scaled(vq_data3)))
        except:
            x.append(0)
        try:
            x.append(math.exp(hmm4.backward_scaled(vq_data4)))
        except:
            x.append(0)

        if (x.index(max(x))==0 and output=="Prostrate Cancer"):
            print "Matched Prostrate Cancer"
            num_correct+=1
        elif (x.index(max(x))==1 and output=="Breast Cancer"):
            print "Matched Creast Cancer"
            num_correct+=1
        elif (x.index(max(x))==2 and output=="Lung Cancer"):
            print "Matched Lung Cancer"
            num_correct+=1
        elif (x.index(max(x))==3 and output=="Colorectal Cancer"):
            print "Matched Colorectal Cancer"
            num_correct+=1

        print x

print num_correct

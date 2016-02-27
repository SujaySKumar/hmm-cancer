import scipy.cluster.vq as sp
import numpy as np
import os, pickle
from myhmm_scaled import MyHmmScaled as HMM
from myhmm import MyHmm

def obtain_training_data(trng_path="./training"):
	trng_data = {}
	vs = {}								# Vector Sizes Mapping
	dis_files = os.listdir(trng_path)
	for df in dis_files:
		disease = df.split(".")[0]
		trng_data[disease] = []
		f = open(trng_path+'/'+df,"r")
		l = f.readlines()
		f.close()
		for i in range(len(l)):
			s1 = l[i][:-2]
			s2 = s1.replace("null","0")
			trng_data[disease].append(map(float, s2.split(",")[1:]))
		d_len = len(trng_data[disease][0])
		if not vs.has_key(d_len):
			vs[d_len]= []
		vs[d_len].append(disease)
	pickle.dump(vs, open("size_mapping.pkl","wb"))
	return trng_data, vs

def vector_quantize(data_dict, vs, bins):
	codebooks = {}
	vq_data = {}
	for size in vs.keys():
		all_size_data = []
		for disease in vs[size]:
			all_size_data.extend(data_dict[disease])
		codebooks[size] = sp.kmeans(np.asarray(all_size_data), bins)[0]
	pickle.dump(codebooks,open("all_codebooks.pkl","wb"))
	for dis in data_dict.keys():
		n = len(data_dict[dis])
		m = len(data_dict[dis][0])
                print "n=",n
                print "m=", m
                #print type(data_dict[dis])
                #print data_dict[dis]
                #print dis
                #exit(0)

		vq_data[dis] = map(str,sp.vq(np.reshape(data_dict[dis],(n,m)), codebooks[len(data_dict[dis][0])])[0])
	return vq_data

if __name__=="__main__":
	bins = 16
	trng, vec_sizes = obtain_training_data()
	vq_data = vector_quantize(trng, vec_sizes, bins)
        #print vq_data

        hmm = HMM(os.path.join(".","initial.json"))
        #print tuple(vq_data['Prostate Cancer'])
        #for i in range(40):

        '''hmm.forward_backward_multi_scaled([vq_data['Prostate Cancer']])
                #s = sum(hmm.pi.values())
                #print s
        #hmm.forward_backward_multi_scaled(('Heads','Tails','Heads'))
        print "A= ", hmm.A
        print "B= ", hmm.B
        print "pi= ", hmm.pi
'''
        import pickle
        ''''f = open("prostate_trained.pkl","wb")
        pickle.dump(hmm, f)
        f.close()'''
        f = open("colorectal_trained.pkl","rb")
        #print pickle.load(f).pi

        hmm=pickle.load(f)
        x = hmm.forward_scaled(['0','2','0','3','1'])
        import math
        print math.exp(x)

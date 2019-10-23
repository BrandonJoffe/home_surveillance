#The given pickel file does not work on a Windows machine
#the following code will make it run well on Windows system as well 

# path to the pkl file i.e the classifier
pkl_file="system/generated-embeddings/classifier.pkl"

#read pickle file line by line
a=open(pkl_file,"rb").readlines() 

# replace \r\n with \n so that it works in Windows
a=map(lambda x:x.replace("\r\n","\n"),a) 

#write back to file in binary mode
with open(pkl_file,"wb") as j: 
    for i in a:
        j.write(i)         

#Load the pickel file
pickle.load(open(pkl_file,"rb")) 

#Now it works well on Windows as well!

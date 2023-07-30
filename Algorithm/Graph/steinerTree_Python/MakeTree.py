import numpy as np

# Random UDG

n = 8 # numNode 
totalCount = 0 # numEdge

maxCount = 3
randomVal = 0
arbi = np.zeros([n,n])
for i in range(n):
    count = 0
    
    for j in range(n):
        if(arbi[i,j] != 0):
            count+=1
    
    for j in range(n):
        if(j == i):
            arbi[i,j] = 0
            
        elif (count < 3):
            randomVal = np.random.uniform() * 1
            
            if(arbi[i,j] != 0):
                pass
            elif(randomVal > 0.5):
                
                edgeWeight = np.random.uniform() * 1 
                #Transpose
                arbi[i,j] = edgeWeight
                arbi[j,i] = edgeWeight
                
                totalCount+=1
                count+= 1
   
            
          
for i in range(n):
    for j in range(n):
        
        if(arbi[i,j] != 0):
            if(j > i):
                print("E", i + 1,j + 1,arbi[i,j])
                
for i in range(n):
    print("W", i + 1)
    
print("R", 1)    

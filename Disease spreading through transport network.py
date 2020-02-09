#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import time
import os
import random
import pandas as pd
import numpy as np
import networkx as nx
import bisect 


# In[12]:


#### a function to visulize the air tranport network
def plot_network_usa(net, xycoords, edges=None, linewidths=None):
    """
    Plot the network usa.
    The file US_air_bg.png should be located in the same directory
    where you run the code.

    Parameters
    ----------
    net : the network to be plotted
    xycoords : dictionary of node_id to coordinates (x,y)
    edges : list of node index tuples (node_i,node_j),
            if None all network edges are plotted.
    linewidths : a list with equal length and order to egdes -list.
            See nx.draw_networkx documentation

    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 0.9])
    # ([0, 0, 1, 1])
    bg_figname = 'US_air_bg.png'
    img = plt.imread(bg_figname)
    axis_extent = (-6674391.856090588, 4922626.076444283,
                   -2028869.260519173, 4658558.416671531)
    ax.imshow(img, extent=axis_extent)
    ax.set_xlim((axis_extent[0], axis_extent[1]))
    ax.set_ylim((axis_extent[2], axis_extent[3]))
    ax.set_axis_off()
    nx.draw_networkx_nodes(net,
                     pos=xycoords,
                     with_labels=False,
                     node_color='k',
                     node_size=5,
                     alpha=0.2)
    if linewidths == None:
        linewidths = np.ones(len(edges))

    for edge, lw  in zip(edges, linewidths):
        nx.draw_networkx_edges(
            net,
            pos=xycoords,
            with_labels=True,
            edge_color='r',
            width=lw,
            edgelist=[edge],
            alpha=lw,
        )
    return fig, ax


# In[13]:


##### visulize the US airport transport network based on the imported data
id_data = np.genfromtxt("US_airport_id_info.csv", delimiter=',', dtype=None, names=True)
xycoords = {}
for row in id_data:
    xycoords[str(row['id'])] = (row['xcoordviz'], row['ycoordviz'])

net = nx.read_weighted_edgelist("aggregated_US_air_traffic_network_undir.edg")

edges=net.edges()

plot_network_usa(net, xycoords, edges)


# In[21]:


# load the event data and sort the order by startime
event=np.genfromtxt("events_US_air_traffic_GMT.txt",names=True,dtype=int)
event.sort(order='StartTime')
event_sort=event


# In[22]:


############# Susceptible-Infected  model function #############
#In the SI model, each node is either Susceptible or Infected. When an Infected node is in contact with a 
#Susceptible node,the Susceptible node may become infected with some probability p ∈ [0,1], reﬂecting the 
# infectivity of the disease. Infected nodes remain Infected forever.

def si_model (id,p,time):
    infection_time={}
    ### set up the seed node and its infection time. the infection time is the first flight's departure time
    infection_time[id]=event[event['Source']==id]['StartTime'].min()
    
    
    #check each event, whether the source is affected, if it is affected, so this flight will take the diease.
    n=event_sort.shape[0]
    
    for i in range(n):
        if event_sort[i]['Source'] in infection_time.keys() and infection_time[event_sort[i]['Source']]<=event_sort[i]['StartTime']:
            if event_sort[i]['Destination'] not in infection_time.keys():
                d=np.random.random_sample()
                if d<=p:
                    infection_time[event_sort[i]['Destination']]=event_sort[i]['EndTime']
            else:
                if event_sort[i]['EndTime']<infection_time[event_sort[i]['Destination']]:
                    infection_time[event_sort[i]['Destination']]=event_sort[i]['EndTime']
    
    ### calculate the number of infected nodes as a functin of time
    infection_list=[]
    infection_list=list(infection_time.values())
    infection_list.sort()

    ### calculate the fraction in the specific time
    n=len(time)
    fraction=np.zeros(n)       ###store the result of specific time for each run
    
    for i in range(n):
        #fraction[i]=list(filter(lambda j:j>time[i],infection_list))[0]
        fraction[i]=bisect.bisect_left(infection_list, time[i])    
        
    return infection_time,infection_list,fraction
    


# In[23]:


timeline=np.linspace(event_sort["StartTime"].min(),event_sort["EndTime"].max(),num=20)


# In[24]:


# task 1-If Allentown (node-id=0) is infected at the beginning of the data set, at which time does Anchorage (ANC, node-id=41) become infected? 
i, j,k=si_model(0,1,timeline)
i[41]


# In[25]:


######## task 2 - effect of infection probability p on spreading speed #########
p_list=[0.01,0.05,0.1,0.5,1.0]

# store the results for each probability
    
result_all=np.zeros([5,len(timeline)])
for i in p_list:
    temp=np.zeros([10,len(timeline)]) ### sotre the temporal result from each run
    
    for j in range(10):
        n,m,k=si_model(0,i,timeline)
        temp[j]=k    
        
    result_all[p_list.index(i)]=np.mean(temp,axis=0)

m=net.number_of_nodes()
r_fraction=result_all/float(m)


# In[26]:


### visualize the resutl of task2

fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle("Effect of infection probability p on spreading speed")

for value,name,color in zip([r_fraction[0],r_fraction[1],r_fraction[2],r_fraction[3],r_fraction[4]],
           ["p=0.01","p=0.05","p=0.1","p=0.5","p=1"],["r", "y", "b", "k","g"]):
    ax.plot(timeline,value,color=color,label=str(name))
    
    
ax.set_ylabel("averaged fraction of infected nodes")
ax.set_xlabel("time")
ax.set_ylim([0.0,1.1])
ax.legend(loc=0)
plt.tight_layout
fig.savefig("./task2")


# In[26]:


###### task 3 Effect of seed node selection on spreading speed ###########
seed_all=np.zeros([5,len(timeline)])

seed=[0,4,41,100,200]

for i in seed:
    temp=np.zeros([10,len(timeline)]) ### sotre the temporal result from each run

    for j in range(10): 
            n,m,k=si_model(i,0.1,timeline)
            temp[j]=k    
    seed_all[seed.index(i)]=np.mean(temp,axis=0)   


m=net.number_of_nodes()
s_fraction=seed_all/float(m)
print(np.round(s_fraction,2))


# In[25]:


######task 3, plot the lines
fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle("Effect of seed node selection on spreading speed")

for value,name,color in zip([s_fraction[0],s_fraction[1],s_fraction[2],s_fraction[3],s_fraction[4]],
           ["ID=0","ID=4","ID=41","ID=100","ID=200"],["r", "y", "b", "k","g"]):
    ax.plot(timeline,value,color=color,label=str(name))
    
    
ax.set_ylabel("averaged fraction of infected nodes")
ax.set_xlabel("time")
ax.set_ylim([0.0,1.1])
ax.legend(loc=0)
plt.tight_layout
fig.savefig("./task3")


# In[ ]:


###########task 4- where to hide


# In[8]:


##############task 4 - find the median for each node ######
m=net.number_of_nodes()
temp_node=np.zeros([m,50])
nodes = list(net.nodes())

# minimun the calculation of timeline, so make it small 
timeline=np.linspace(event_sort["StartTime"].min(),event_sort["EndTime"].min(),num=2)


for n in range(50):
    seed=int(np.random.choice(nodes))
    i,j,k=si_model(seed,0.5,timeline)
    for key,value in i.items():
        temp_node[int(key),n]=value


# In[9]:


#########task 4 calculate the median, degree, center, coefficient
n_data=np.zeros([m,5])
### find the median and fix the value in first colunm
n_data[:,0]=np.median(temp_node,axis=1)

###calculate the unweighted clustering coefficient c
c=nx.clustering(net)
for key, value in c.items():
    n_data[int(key),1]=value

###calulate the degree k 
k=net.degree()
for i in k:
    n_data[int(i[0]),2]=i[1]

#######calculate the strength s
s=nx.degree(net,weight="weight")
for i in s:
    n_data[int(i[0]),3]=i[1]

########calculate the betweenness
b=nx.betweenness_centrality(net)
for i,j in b.items():
    n_data[int(i),4]=j

print(n_data)


# In[20]:


import operator
sorted(c.items(),key=operator.itemgetter(1),reverse=True)


# In[10]:


############task 4 visulize the data

for x_value,color,x_label,name in zip([n_data[:,1],n_data[:,2],n_data[:,3],n_data[:,4]],["r", "y", "k","g"],
                                 ["unweighted clustering coefficient c","degree k","strength s","unweighted betweenness centrality"],
                                     ["unweighted clustering coefficient c","degree k","strength s","unweighted betweenness centrality"]):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x_value,n_data[:,0],marker="o",c=color,alpha=0.3)
    ax.set_xlabel(x_label)
    ax.set_ylabel("median infection time ")
    ax.grid()
    fig.savefig("./task4_"+name)


# In[11]:


######task 4  calculate the spearman rank-correlation coefficient
from scipy import stats
sp_c=stats.spearmanr(n_data[:,0],n_data[:,1])
sp_k=stats.spearmanr(n_data[:,0],n_data[:,2])
sp_s=stats.spearmanr(n_data[:,0],n_data[:,3])
sp_b=stats.spearmanr(n_data[:,0],n_data[:,4])

print("unweighted clustering coefficient c",sp_c)
print("degree k",sp_k)
print("strength s",sp_s)
print("unweighted betweenness centrality",sp_b)


# In[38]:


print("Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. Correlations of -1 or +1 imply an exact monotonic relationship. Positive correlations imply that as x increases, so does y. Negative correlations imply that as x increases, y decreases.")


# In[ ]:





# In[21]:


#########task 5-step1  list all the immunization nodes of 6 strategies

#### strategy 1_ find the neighbor of a random node
immu_neigh=[]
nodes=list(net.nodes())
i=0
while len(immu_neigh)<10:
    r_node=np.random.choice(nodes)
    neighs=list(net.neighbors(r_node))
    neigh=int(np.random.choice(neighs))
    if neigh not in immu_neigh:
        immu_neigh.append(neigh)
        
#### strategy 2_ find the random node   
immu_node=[]
while len(immu_node)<10:
    r_node=int(np.random.choice(nodes))
    if r_node not in immu_node:
        immu_node.append(r_node)

#### strategy 3- find the clustering coefficient c
immu_c=[]
idx_c=(-n_data[:,1]).argsort()[:10]  ### sort and select the maximum 10 values' index
immu_c=list(idx_c)

#### strategy 4-degree k
immu_k=[]
idx_k=(-n_data[:,2]).argsort()[:10] 
immu_k=list(idx_k)

### strategy 5-strength s
immu_s=[]
idx_s=(-n_data[:,3]).argsort()[:10] 
immu_s=list(idx_s)

### strategy 6-betweenness centrality b
immu_b=[]
idx_b=(-n_data[:,4]).argsort()[:10] 
immu_b=list(idx_b)


# In[22]:


#########task 5-step 2  select the 20 random seed nodes

#### combine all the immunization nodes into a list
immu_all=[]
immu_all.append(immu_neigh)
immu_all.append(immu_node)
immu_all.append(immu_c)
immu_all.append(immu_k)
immu_all.append(immu_s)
immu_all.append(immu_b)

#### select the seed nodes
seed_20=[]
while len(seed_20)<20:
    r_node=int(np.random.choice(nodes))
    if r_node not in immu_all:
        seed_20.append(r_node)


# In[23]:


##### task 5-step 3:write a new si_model as a new funciton
def si_model_immu (id,p,time,immu):
    infection_time={}
    ### set up the seed node and its infection time. the infection time is the first flight's departure time
    infection_time[id]=event[event['Source']==id]['StartTime'].min()
    
    
    #check each event, whether the source is affected, if it is affected, so this flight will take the diease.
    n=event_sort.shape[0]  ### all the event
    
    for i in range(n):
        if event_sort[i]['Source'] in infection_time.keys() and infection_time[event_sort[i]['Source']]<=event_sort[i]['StartTime']:
            if event_sort[i]['Destination'] not in immu:
                if event_sort[i]['Destination'] not in infection_time.keys():
                    d=np.random.random_sample()
                    if d<p:
                        infection_time[event_sort[i]['Destination']]=event_sort[i]['EndTime']
                else:
                    if event_sort[i]['EndTime']<infection_time[event_sort[i]['Destination']]:
                        infection_time[event_sort[i]['Destination']]=event_sort[i]['EndTime']
            
    ### calculate the number of infected nodes as a functin of time
    infection_list=[]
    infection_list=list(infection_time.values())
    infection_list.sort()

    ### calculate the fraction in the specific time
    n=len(time)
    fraction=np.zeros(n)       ###store the result of specific time for each run
    
    for i in range(n):
        #fraction[i]=list(filter(lambda j:j>time[i],infection_list))[0]
        fraction[i]=bisect.bisect_left(infection_list, time[i])    
    m=net.number_of_nodes()
    fraction1=fraction/float(m)
    
    return infection_time,infection_list,fraction1
    


# In[24]:


##### task 5- step 4: simulate 20 times of the model for each strategy

timeline5=np.linspace(event_sort["StartTime"].min(),event_sort["EndTime"].max(),num=20)
immu_fra=np.zeros([6,len(timeline5)])

for j,immu in enumerate(immu_all):
    temp=np.zeros([20,len(timeline5)]) ### rows:20 runs, column: fraction of 20 time-intervals
    for i,seed in enumerate(seed_20):
        n,m,k=si_model_immu(seed,0.5,timeline5,immu)
        temp[i]=k
    immu_fra[j]=np.mean(temp,axis=0)
    


# In[25]:


### task 5-step 5:  visualize the resutl 

fig = plt.figure(figsize=(16, 16 * 3 / 4.))
ax = fig.add_subplot(111)
fig.suptitle("shutting down airport")

for value,name,color in zip([immu_fra[0],immu_fra[1],immu_fra[2],immu_fra[3],immu_fra[4],immu_fra[5]],
           ["random neighbour of random node","random node","unweighted clustering coefficient c","degree k",
            "strength s","unweighted betweenness centrality"],["r", "y", "b", "k","g","m"]):
    ax.plot(timeline5,value,color=color,label=str(name))

ax.set_ylabel("averaged fraction of infected nodes")
ax.set_xlabel("time")
ax.set_ylim([0.0,1.1])
ax.legend(loc=2)
plt.tight_layout
fig.savefig("./task5")


# In[8]:


#### task 6-step1 create a function to store the links information
def si_model_link (id,p):
    infection_time={}
    m=len(net.nodes())
    infection_link=np.zeros([m,m])
    ### set up the seed node and its infection time. the infection time is the first flight's departure time
    infection_time[id]=event[event['Source']==id]['StartTime'].min()
    
    
    #check each event, whether the source is affected, if it is affected, so this flight will take the diease.
    n=event_sort.shape[0]
    
    for i in range(n):
        if event_sort[i]['Source'] in infection_time.keys() and infection_time[event_sort[i]['Source']]<=event_sort[i]['StartTime']:
            if event_sort[i]['Destination'] not in infection_time.keys():
                d=np.random.random_sample()
                if d<=p:
                    infection_time[event_sort[i]['Destination']]=event_sort[i]['EndTime']
                    infection_link[event_sort[i]['Source'],event_sort[i]['Destination']]=1
            else:
                if event_sort[i]['EndTime']<infection_time[event_sort[i]['Destination']]:
                    infection_time[event_sort[i]['Destination']]=event_sort[i]['EndTime']
                    infection_link[:,event_sort[i]['Destination']]=0
                    infection_link[event_sort[i]['Source'],event_sort[i]['Destination']]=1
    
   
       
    return infection_time,infection_link
    


# In[ ]:


########task 6- step 2: record the links which bring infection and store it into dictionary (link:weight)
m=len(net.nodes())
rec_20=np.zeros([m,m])  ### rows: sources, column: destination, value: sum of all the record in 20 simulation.
nodes=list(net.nodes())

for i in range(20):
    seed=int(np.random.choice(nodes))
    time,link=si_model_link(seed,0.5)
    rec_20+=link

rec_20_fra=rec_20/20
edge_fra={}
for i in range(m):
    for j in range(m):
        if (rec_20_fra[i,j] !=0:
        edge_fra[(str(i),str(j))]=rec_20_fra[i,j]
      


# In[ ]:





# In[33]:


######### task 6 visulize the result

lw_list =[]
edgelist =[]
for key,value in edge_fra.items():
    lw_list.append(value) # change this at least
    edgelist.append(key) # edgelist created to maintain the right order

fig,ax=plot_network_usa(net, xycoords, edges=edgelist, linewidths=lw_list)
fig.savefig("./task6-network")


# In[52]:


##### task 6 - fij as a function of the link properties
 
wij=list(net.edges(data=True))
ebij=nx.edge_betweenness_centrality(net)
fij=edge_fra 

n=net.number_of_edges()

e_val=np.zeros([n,5])  ### row: links , column:(0-node 1), (1-node 2),(2-wij),(3-ebij),(4-fij)

for i in range(n):
    e_val[i,0]=int(wij[i][0])
    e_val[i,1]=int(wij[i][1])
    e_val[i,2]=wij[i][2]["weight"]
    e_val[i,3]=ebij[(str(int(e_val[i,0])),str(int(e_val[i,1])))]
    if e_val[i,0]<e_val[i,1]:
        e_val[i,4]=fij[(str(int(e_val[i,0])),str(int(e_val[i,1])))]
    else:
        e_val[i,4]=fij[(str(int(e_val[i,1])),str(int(e_val[i,0])))]

print(e_val)


# In[53]:


#####task 6 visualize the result
for x_value,color,x_label in zip([e_val[:,2],e_val[:,3]],["r", "y"],
                                 ["link weight wij","unweighted link betweenness centrality ebij"]):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x_value,e_val[:,4],marker="o",c=color,alpha=0.3)
    ax.set_xlabel(x_label)
    ax.set_ylabel("fraction of times ")
    ax.grid()
    fig.savefig("./task6_"+x_label)


# In[54]:


######task 6  calculate the spearman rank-correlation coefficient
from scipy import stats
sp_wij=stats.spearmanr(e_val[:,4],e_val[:,2])
sp_ebij=stats.spearmanr(e_val[:,4],e_val[:,3])


print("link weight wij",sp_wij)
print("unweighted link betweenness centrality ebij",sp_ebij)


# In[ ]:


####### task 7- find out an appropriate measure to predict fij.


# In[114]:


#########task 7 - create the array for different test.
t_val=np.zeros([n,9]) ### row: all links, column:(0-node1),(1-node2),(2-fij),(3-6 for different test)
t_val[:,:2]=e_val[:,:2]
t_val[:,2]=e_val[:,4]


# In[80]:


#########task 7 - unweighted links selected from maximum spanning tree

Maximum=nx.maximum_spanning_tree(net)
max_edge=Maximum.edges()

for i in range(n):
    if (str(int(t_val[i,0])),str(int(t_val[i,1]))) in max_edge:
        t_val[i,3]=1
        
sp_max=stats.spearmanr(t_val[:,2],t_val[:,3])
print("unweighted links selected from maximum spanning tree",sp_max)


# In[82]:


#########task 7 - weighted links selected from maximum spanning tree

Maximum=nx.maximum_spanning_tree(net)
max_edge=Maximum.edges(data=True)

for i in range(n):
    if t_val[i,3]==1:
        for t,q in enumerate(max_edge):
            if q[0]==str(int(t_val[i,0])) and q[1]==str(int(t_val[i,1])):
                t_val[i,4]=q[2]["weight"] 

sp_max_w=stats.spearmanr(t_val[:,2],t_val[:,4])
print("weighted links selected from maximum spanning tree",sp_max_w)


# In[83]:


#########task 7 - unweighted links selected from minimum spanning tree

Minimum=nx.minimum_spanning_tree(net)
min_edge=Minimum.edges()

for i in range(n):
    if (str(int(t_val[i,0])),str(int(t_val[i,1]))) in min_edge:
        t_val[i,5]=1
        
sp_min=stats.spearmanr(t_val[:,2],t_val[:,5])
print("unweighted links selected from minimum spanning tree",sp_min)


# In[95]:


#########task 7 - Current-flow betweenness centrality uses an electrical
#current model for information spreading in contrast to betweenness centrality which uses shortest paths.
edge_cu=nx.edge_current_flow_betweenness_centrality(net)

for i in range(n):
    if (str(int(t_val[i,0])),str(int(t_val[i,1]))) in edge_cu.keys():
        t_val[i,6]=edge_cu[(str(int(t_val[i,0])),str(int(t_val[i,1])))]
    if (str(int(t_val[i,1])),str(int(t_val[i,0]))) in edge_cu.keys():
        t_val[i,6]=edge_cu[(str(int(t_val[i,1])),str(int(t_val[i,0])))]

sp_cu=stats.spearmanr(t_val[:,2],t_val[:,6])
print("edge_current_flow_betweenness_centrality",sp_cu)


# In[100]:


#########task 7 - edge_load_centrality
edge_lo=nx.edge_load_centrality(net)
for i in range(n):
    if (str(int(t_val[i,0])),str(int(t_val[i,1]))) in edge_lo.keys():
        t_val[i,7]=edge_lo[(str(int(t_val[i,0])),str(int(t_val[i,1])))]
    if (str(int(t_val[i,1])),str(int(t_val[i,0]))) in edge_lo.keys():
        t_val[i,7]=edge_lo[(str(int(t_val[i,1])),str(int(t_val[i,0])))]

sp_lo=stats.spearmanr(t_val[:,2],t_val[:,7])
print("edge_load_centrality",sp_lo)


# In[116]:


#########task 7 - thresholded networks
edges = net.edges.data()
descending_order = sorted(edges,key=lambda edge: edge[2]["weight"],reverse=True)
M=nx.number_of_edges(Maximum)
thred=descending_order[:M]
thred_list=[]
for i in list(thred):
    thred_list.append((i[0],i[1]))

for i in range(n):
    if (str(int(t_val[i,0])),str(int(t_val[i,1]))) in thred_list:
        t_val[i,8]=1
    if (str(int(t_val[i,1])),str(int(t_val[i,0]))) in thred_list:
        t_val[i,8]=1
        
sp_th=stats.spearmanr(t_val[:,2],t_val[:,8])
print("thresholded networks",sp_th)


# In[ ]:


####### bonus task_ optimize the Simple SI model function, make it more realistic


# In[73]:


############# Bonus task - SI model function #############
def si_model_op(m,id,p,time):
    infection_time={}
    ### set up the seed node and its infection time. the infection time is the first flight's departure time
    infection_time[id]=event[event['Source']==id]['StartTime'].min()
    
    m=net.number_of_edges()
    #check each event, whether the source is affected, if it is affected, so this flight will take the diease.
    n=event_sort.shape[0]
    
    for i in range(n):
        if event_sort[i]['Source'] in infection_time.keys() and infection_time[event_sort[i]['Source']]<=event_sort[i]['StartTime']:
            if event_sort[i]['Destination'] not in infection_time.keys():
                d=np.random.random_sample()
                k= len(list(net.neighbors(str(event_sort[i]['Destination']))))
                e= (p+k/m)/2
                if d<=e:
                    infection_time[event_sort[i]['Destination']]=event_sort[i]['EndTime']+60*60
            else:
                if event_sort[i]['EndTime']+ 60*60 <infection_time[event_sort[i]['Destination']]:
                    infection_time[event_sort[i]['Destination']]=event_sort[i]['EndTime']+60*60
    
    ### calculate the number of infected nodes as a functin of time
    infection_list=[]
    infection_list=list(infection_time.values())
    infection_list.sort()

    ### calculate the fraction in the specific time
    n=len(time)
    fraction=np.zeros(n)       ###store the result of specific time for each run
    
    for i in range(n):
        #fraction[i]=list(filter(lambda j:j>time[i],infection_list))[0]
        fraction[i]=bisect.bisect_left(infection_list, time[i])    
        
    return infection_time,infection_list,fraction
    


# In[74]:


## bonus task , do task 2 again.
p_list=[0.01,0.05,0.1,0.5,1.0]
 ### store the results for each probability
    
result_all=np.zeros([5,len(timeline)])
for i in p_list:
    temp=np.zeros([10,len(timeline)]) ### sotre the temporal result from each run
    
    for j in range(10):
        n,m,k=si_model_op(m,0,i,timeline)
        temp[j]=k    
        
    result_all[p_list.index(i)]=np.mean(temp,axis=0)

m=net.number_of_nodes()
r_fraction=result_all/float(m)
print(np.round(r_fraction,2))


# In[75]:


fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle("Effect of infection probability p on spreading speed with opitimized model")

for value,name,color in zip([r_fraction[0],r_fraction[1],r_fraction[2],r_fraction[3],r_fraction[4]],
           ["p=0.01","p=0.05","p=0.1","p=0.5","p=1"],["r", "y", "b", "k","g"]):
    ax.plot(timeline,value,color=color,label=str(name))
    
    
ax.set_ylabel("averaged fraction of infected nodes")
ax.set_xlabel("time")
ax.set_ylim([0.0,1.1])
ax.legend(loc=0)
plt.tight_layout
fig.savefig("./task10")


# In[48]:


###### bonus task  Effect of seed node selection on spreading speed ###########
seed_all=np.zeros([5,len(timeline)])

seed=[0,4,41,100,200]

for i in seed:
    temp=np.zeros([10,len(timeline)]) ### sotre the temporal result from each run

    for j in range(10): 
            n,m,k=si_model_op(m,i,0.1,timeline)
            temp[j]=k    
    seed_all[seed.index(i)]=np.mean(temp,axis=0)   


m=net.number_of_nodes()
s_fraction=seed_all/float(m)
print(np.round(s_fraction,2))


# In[50]:


######bonus-task, plot the lines
fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle("Effect of seed node selection on spreading speed")

for value,name,color in zip([s_fraction[0],s_fraction[1],s_fraction[2],s_fraction[3],s_fraction[4]],
           ["ID=0","ID=4","ID=41","ID=100","ID=200"],["r", "y", "b", "k","g"]):
    ax.plot(timeline,value,color=color,label=str(name))
    
    
ax.set_ylabel("averaged fraction of infected nodes")
ax.set_xlabel("time")
ax.set_ylim([0.0,1.1])
ax.legend(loc=0)
plt.tight_layout
fig.savefig("./task9")


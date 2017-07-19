#coding:utf-8
from  random import randint

def str_generator(start,end,scopes=[(65,90),(97,122)]):
    """
        默认生成scopes区间的字符串
    """
    rand_str=[] #随机字符列表
    n=1        #计数
    len_scopes=len(scopes) #范围的长度
    num=randint(start,end) #生成start-end之间任意长度字符串
    for x in range(num):
        index=randint(0,len_scopes-1)    #随机选择 a-z 或 A-Z 区间
        rand_int=randint(*scopes[index]) #生成该区间之间的随机数,包含起点和终点
        rand_str.append(chr(rand_int))
        print "%.f%%    "%(n*100.0/num)+"##"*n #进度
        n+=1
    return "".join(rand_str)



#example1:
# 默认a-z  A-Z范围
print "the generate string is: %s"%str_generator(20,30)    

#example2:
# #ascii  0-100
# print "the generate string is: %s"%str_generator(1,10,[(0,100)])

#example3:
# #ascii  0-255      
# print "the generate string is: %s"%str_generator(1,10,[(0,100)])

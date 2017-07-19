#coding:utf-8
import time

def expensiveFunc(infile,outfile="output.txt"):
    try:
        lines=open(infile,'rt').readlines() #读取文本内容
        LEN=len(lines) #内容行数
        n=1    #计数
        with open(outfile,'wt') as f:
            for line in lines:
                line=line.strip().replace(" ","")                #去除空格
                sorted_strs="".join(sorted(line)) #分割字符串 并排序
                f.write(sorted_strs+"\n")  #写出一行
                time.sleep(0.001)     #延迟0.01s
                print "%.f%%    "%(n*100.0/LEN)+"####"*n #进度
                n+=1
    except IOError as e:
        print "Error:"+e
    else:
        print "Done!"

        
expensiveFunc("input.txt")








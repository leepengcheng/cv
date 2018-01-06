#coding:utf-8
from turtle import *
from PIL import Image
import os
import yaml
import numpy as np


class Window():
    def __init__(self,map_png_dir,map_yaml_dir):
        map_name=os.path.basename(map_png_dir) ##xxx.png
        png = Image.open(map_png_dir)
        self.pix=png.load() #PixelAccess
        self.size = png.size  #地图png尺寸,注意和numpy.asarray颠倒过来
        gif_name = "%s.gif" % map_name.split(".")[0]
        setup(*self.size)           #设置宽和高
        bgpic(gif_name)      #设置背景为地图
        speed(1)             #设置仿真速度
        shape("turtle")      #形状
        ##复制gif到本目录
        if not os.path.exists(gif_name):
            #转换为gif
            gif = png.convert('RGB').convert('P', palette=Image.ADAPTIVE)
            gif.save(gif_name)
            gif.close()
        cfg=yaml.load(open(map_yaml_dir)) 
        # origin=cfg['origin']  #地图起点
        # res=cfg['resolution'] #每个像素代表的长度m
        self.occ=cfg['occupied_thresh'] #占用网格阈值:occ=(255-color_ave)/255,大于occ认为占用
        self.free=cfg['free_thresh']   #自由网格阈值:free=(255-color_ave)/255,小于free认为自由
        # cfg['negate']    #白/黑自由/占用语义是否应该被反转
        self.screen=Screen()
        self.screen.onclick(self.print_pixstate)
    

    def print_pixstate(self,x,y):
        "获得像素点的状态"
        w,h=self.size
        print "Center Origin XY :%s,%s"%(x,y)
        x=x+w/2
        y=h/2-y
        print "LeftTop Origin XY :%s,%s"%(x,y)
        state=1
        if 0<=x<w and 0<=y<h:
            color_ave=sum(self.pix[x,y][:3])/3.0
            ratio=(255-color_ave)/255.0
            if ratio>=self.occ:
                state=1
            elif ratio<=self.free:
                state=0
            else:
                state=-1
        print "Point State:%s"%state

        # def get_pixstate(self,x,y):
        #     '''
        #     true时原点在左下方
        #     false时原点在左上方
        #     '''
        #     global w,h,pix,occ,free
        #     y=h-y
        #     if 0<=x<w and 0<=y<h:
        #         color_ave=sum(self.pix[x,y][:3])/3.0
        #         ratio=(255-color_ave)/255.0
        #         if ratio>=occ:
        #             return 1
        #         elif ratio<=free:
        #             return 0
        #         else:
        #             return -1
        #     #外部全部占用
        #     return 0







#     #初始化界面
#     init_form()
#     ############
#     start=100,100,45
#     goal=550,450,135
#     move_to(*start,showpath=False)
#     move_to(*goal,v=1)

#     graph={}
#     lin=(-1,0,1)
#     for x in xrange(w):
#         for y in xrange(h):
#             #获得(x,y)处像素的状态 占用/自由/未知 1/0/-1
#             subnodes={}
#             for i in lin:
#                 for j in lin:
#                     if i==j==0:
#                         continue
#                     else:
#                         pos=(x+i,y+j)
#                         if get_pixstate(*pos)==0:
#                             subnodes[pos]=10+abs(i*j)*4
#             graph[(x,y)]= subnodes




# def move_to(x,y,angle,v=0,showpath=True):
#     global w,h
#     speed(v)
#     if not showpath:
#         penup()
#     goto(x-w/2,y-h/2)
#     seth(angle)
#     pendown()


#地图和配置文件的路径
map_dir = r"H:\RepoWindows\stdr_simulator-indigo-devel\stdr_resources\maps"
map_name = "sparse_obstacles.png"  #地图png
yaml_name = "sparse_obstacles.yaml"  #地图yaml
map_png_dir = os.path.join(map_dir, map_name)
map_yaml_dir = os.path.join(map_dir, yaml_name)
win=Window(map_png_dir,map_yaml_dir)






#######主循环
mainloop()
import random
import numpy as np
import copy

class Scheduling:
    def __init__(self, n_1,n_2,J_num,Agv_num,Cq_num,Gui_num,zhuang,sigma_):
        self.J_num = J_num
        self.n_1 = n_1
        self.n_2 = n_2
        self.Agv_num=Agv_num
        self.Cq_num=Cq_num
        self.Gui_num=Gui_num
        self.zhuang=zhuang
        self.sigma= sigma_
        self.alpha = [0.0333, 0.0233, 0.0183]
        self.beta = [0.0666, 0.0266, 0.0166]
        self.lamda=[0.5,0.5]
        self.Fit_E=0
        self.F_1=[]
        self.T_1 = [0 for _ in range(Agv_num)]
        self.T_2 = [0 for _ in range(Gui_num)]
        # self.Agv_w1 = [1 for _ in range(Agv_num)]
        # self.Agv_w2 = [21, 25, 29]
        # self.Gui_w1 = [2 for _ in range(Agv_num)]
        # self.Gui_w2 = sorted(random.sample(range(0, 50), Gui_num))
        # self.Agv_w = [list(t) for t in zip(self.Agv_w1, self.Agv_w2)]
        # self.Gui_w = [list(t) for t in zip(self.Gui_w1, self.Gui_w2)]
        self.Agv_J = [["-" for col in range(J_num)] for row in range(Agv_num)]
        self.Gui_J = [["-" for col in range(J_num)] for row in range(Gui_num)]
        self.Agv_t = [["-" for col in range(J_num)] for row in range(Agv_num)]
        self.Gui_t = [["-" for col in range(J_num)] for row in range(Gui_num)]
        self.T_ah = [0 for _ in range(Agv_num)]
        self.T_gh = [0 for _ in range(Gui_num)]
        self.T_lb1 = [0 for _ in range(4)]
        self.T_lb2 = [0 for _ in range(4)]
        self.T_lb3 = [0 for _ in range(4)]
        self.TT_1 = 0
        self.TT_2 = 0
        self.TT_3 = 0
        self.TTZ = 60
        self.zuidat = 0
        self.Gui_wz=[[21,25,29]]
        self.Gui_tz=[[0,0,0]]
        self.t1=0
        self.t2=0
        self.t3=0
    def shiijan_1(self,z1,z2):
        z = [z1[i] - z2[i] for i in range(len(z1))]
        t_shichang = (800 * abs(z[0]) + 17 * abs(z[1]))/10
        return t_shichang
    def shiijan_2(self,z1,z2):
        z = [z1[i] - z2[i] for i in range(len(z1))]
        t_shichang = (800 * abs(z[0]) + 17 * abs(z[1]))/5
        return t_shichang
    def shiijan_3(self,z1,z2):
        z = [z1[i] - z2[i] for i in range(len(z1))]
        t_shichang = (800 * abs(z[0]) + 17 * abs(z[1]))/(4/3)
        return t_shichang

    def E_fitness(self,eg0,eg1,eg2,ev0,ev1,ev2):
        EG=eg0*self.alpha[0]+eg1*self.alpha[1]+eg2*self.alpha[2]
        Ev=ev0*self.beta[0]+ev1*self.beta[1]+ev2*self.alpha[2]
        E_Fitness=Ev+EG
        return E_Fitness

    def Decode(self, RST):
        Agv_w=[[1,21],[1,25],[1,29],[1,21],[1,25],[1,29]]
        Gui_w=[[2,21],[2,25],[2,29]]
        for i in range(len(RST)):
            keyong_G=[]
            s_2=[]
            s_3 = []
            s_4 = []
            s_bh1 = []
            s_bh2 = []
            a = RST[i]
            T_11 = []
            T_2_=[]
            T_3 = []
            b=0
            cun=[]
            w_n_2=0
            cqblza = []
            cqblzg = []
            cqjs = []
            otiv = 0
            cadeng = 0
            if a <self.zhuang:
               for j in range(self.Agv_num):
                    t_1 = self.T_ah[j] + self.shiijan_1(self.n_1[a], Agv_w[j])
                    self.T_1[j]=t_1
               T_1min = self.T_1.index(min(self.T_1))
               if self.n_1[a][1]==21:
                   if self.T_1[T_1min]<=(self.TT_1+2*self.TTZ):
                       for ii in range(4):
                           tc=self.T_1[T_1min]-self.T_lb1[ii]
                           cqjs.append(tc)
                           if tc>=0:
                               cqblza.append(ii)
                       if cqblza==[]:
                            k2 = cqjs.index(max(cqjs))


                            if (self.TT_1+2*self.TTZ)>=self.T_lb1[k2]:
                                self.TT_1= self.TT_1+2*self.TTZ
                                self.T_lb1[k2]=self.TT_1
                                otiv = self.TT_1 + self.shiijan_2(self.n_1[a], self.n_2[a])
                                cadeng = self.TT_1-self.T_1[T_1min]
                            else:
                                self.TT_1 = self.T_lb1[k2]

                                otiv = self.T_lb1[k2] + self.shiijan_2(self.n_1[a], self.n_2[a])
                                cadeng = self.T_lb1[k2] - self.T_1[T_1min]
                       else:
                           k=cqblza[0]

                           self.TT_1 = self.TT_1 + 2*self.TTZ
                           self.T_lb1[k]=self.TT_1
                           otiv = self.TT_1 + self.shiijan_2(self.n_1[a], self.n_2[a])

                   else:
                       for ii in range(4):
                           tc=self.TT_1+2*self.TTZ-self.T_lb1[ii]
                           cqjs.append(tc)
                           if tc>=0:
                               cqblzg.append(ii)
                       if cqblzg==[]:
                           k2 = cqjs.index(max(cqjs))


                           if self.T_lb1[k2]>=self.T_1[T_1min]:
                               self.TT_1= self.T_lb1[k2]

                               otiv = self.T_lb1[k2] + self.shiijan_2(self.n_1[a], self.n_2[a])
                               cadeng = self.T_lb1[k2] - self.T_1[T_1min]
                           else:
                               self.TT_1 = self.T_lb1[k2]
                               self.T_lb1[k2] =  self.T_1[T_1min]
                               otiv = self.T_1[T_1min] + self.shiijan_2(self.n_1[a],
                                                                      self.n_2[a])

                       else:
                           k1 = cqblzg[0]

                           self.TT_1 = self.TT_1+ 2*self.TTZ
                           self.T_lb1[k1]=self.T_1[T_1min]
                           otiv = self.T_1[T_1min] + self.shiijan_2(self.n_1[a], self.n_2[a])

               if self.n_1[a][1]==25:
                   if self.T_1[T_1min]<=(self.TT_2+2*self.TTZ):
                       for ii in range(4):
                           tc=self.T_1[T_1min]-self.T_lb2[ii]
                           cqjs.append(tc)
                           if tc>=0:
                               cqblza.append(ii)
                       if cqblza==[]:
                            k2 = cqjs.index(max(cqjs))

                            if (self.TT_2+2*self.TTZ)>=self.T_lb2[k2]:
                                self.TT_2= self.TT_2+2*self.TTZ
                                self.T_lb2[k2]=self.TT_2
                                otiv = self.TT_2 + self.shiijan_2(self.n_1[a], self.n_2[a])
                                cadeng = self.TT_2-self.T_1[T_1min]
                            else:
                                self.TT_2 = self.T_lb2[k2]

                                otiv = self.T_lb2[k2] + self.shiijan_2(self.n_1[a], self.n_2[a])
                                cadeng = self.T_lb2[k2] - self.T_1[T_1min]
                       else:
                           k=cqblza[0]

                           self.TT_2 = self.TT_2 + 2*self.TTZ
                           self.T_lb2[k]=self.TT_2
                           otiv = self.TT_2 + self.shiijan_2(self.n_1[a], self.n_2[a])
                   else:
                       for ii in range(4):
                           tc=self.TT_2+2*self.TTZ-self.T_lb2[ii]
                           cqjs.append(tc)
                           if tc>=0:
                               cqblzg.append(ii)
                       if cqblzg==[]:
                           k2 = cqjs.index(max(cqjs))

                           if self.T_lb2[k2]>=self.T_1[T_1min]:
                               self.TT_2= self.T_lb2[k2]
                               otiv = self.T_lb2[k2] + self.shiijan_2(self.n_1[a], self.n_2[a])
                               cadeng = self.T_lb2[k2] - self.T_1[T_1min]
                           else:
                               self.TT_2 = self.T_lb2[k2]
                               self.T_lb2[k2] =  self.T_1[T_1min]
                               otiv = self.T_1[T_1min] + self.shiijan_2(self.n_1[a],
                                                                      self.n_2[a])

                       else:
                           k1 = cqblzg[0]

                           self.TT_2 = self.TT_2+ 2*self.TTZ
                           self.T_lb2[k1]=self.T_1[T_1min]
                           otiv = self.T_1[T_1min] + self.shiijan_2(self.n_1[a], self.n_2[a])

               if self.n_1[a][1]==29:
                   if self.T_1[T_1min]<=(self.TT_3+2*self.TTZ):
                       for ii in range(4):
                           tc=self.T_1[T_1min]-self.T_lb3[ii]
                           cqjs.append(tc)
                           if tc>=0:
                               cqblza.append(ii)
                       if cqblza==[]:
                            k2 = cqjs.index(max(cqjs))
                            if (self.TT_3+2*self.TTZ)>=self.T_lb3[k2]:
                                self.TT_3= self.TT_3+2*self.TTZ
                                self.T_lb3[k2]=self.TT_3
                                otiv = self.TT_3 + self.shiijan_2(self.n_1[a], self.n_2[a])
                                cadeng = self.TT_3-self.T_1[T_1min]
                            else:
                                self.TT_3 = self.T_lb3[k2]

                                otiv = self.T_lb3[k2] + self.shiijan_2(self.n_1[a], self.n_2[a])
                                cadeng = self.T_lb3[k2] - self.T_1[T_1min]
                       else:
                           k=cqblza[0]
                           self.TT_3 = self.TT_3 + 2*self.TTZ
                           self.T_lb3[k]=self.TT_3
                           otiv = self.TT_3 + self.shiijan_2(self.n_1[a], self.n_2[a])

                   else:
                       for ii in range(4):
                           tc=self.TT_3+2*self.TTZ-self.T_lb3[ii]
                           cqjs.append(tc)
                           if tc>=0:
                               cqblzg.append(ii)
                       if cqblzg==[]:
                           k2 = cqjs.index(max(cqjs))
                           if self.T_lb3[k2]>=self.T_1[T_1min]:
                               self.TT_3= self.T_lb3[k2]
                               otiv = self.T_lb3[k2] + self.shiijan_2(self.n_1[a], self.n_2[a])
                               cadeng = self.T_lb3[k2] - self.T_1[T_1min]
                           else:
                               self.TT_3 = self.T_lb3[k2]
                               self.T_lb3[k2] =  self.T_1[T_1min]
                               otiv = self.T_1[T_1min] + self.shiijan_2(self.n_1[a],
                                                                      self.n_2[a])

                       else:
                           k1 = cqblzg[0]
                           self.TT_3 = self.TT_3+ 2*self.TTZ
                           self.T_lb3[k1]=self.T_1[T_1min]
                           otiv = self.T_1[T_1min] + self.shiijan_2(self.n_1[a], self.n_2[a])

               self.Agv_J[T_1min][i]= a

               for i__ in range(self.Gui_num):
                   if Gui_w[i__][1]<=self.n_2[a][1]:
                       s_bh1.append(i__)
                   else:
                       s_bh2.append(i__)
               if len(s_bh1) and len(s_bh2):
                   k=[s_bh1[-1],s_bh2[0]]
                   for k_i in k:
                       if Gui_w[k_i][1]+1==self.n_2[a][1]:
                           cun.append(k_i)
                       elif Gui_w[k_i][1]-1==self.n_2[a][1]:
                           cun.append(k_i)
                       else:
                           continue
                   if len(cun)==2:
                       if Gui_w[cun[0]][1]-2 != 0:
                           for te in range(self.Gui_num):
                               if Gui_w[cun[0]][1] - 2 == Gui_w[te][1]:
                                   w_n_2=te
                               else:
                                   continue
                           if w_n_2>0:
                               zuida_s = self.T_gh.index(max(self.T_gh))
                               z_time=self.T_gh[zuida_s]
                               Gui_w[cun[0]][1] = Gui_w[cun[0]][1] - 1
                               Gui_w[w_n_2][1] = Gui_w[w_n_2][1] - 1
                               self.T_gh[cun[0]] = z_time + self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                               if cun[0]==0:
                                    self.t1=self.t1+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                               elif cun[0]==1:
                                    self.t2=self.t2+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                               elif cun[0]==2:
                                    self.t3=self.t3+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                               self.T_gh[w_n_2] =z_time+self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1])
                               if w_n_2==0:
                                    self.t1=self.t1+ self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1])
                               elif w_n_2==1:
                                    self.t2=self.t2+ self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1])
                               elif w_n_2==2:
                                    self.t3=self.t3+ self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1])
                               self.T_gh[s_bh2[0]]=z_time
                               keyong_G = [s_bh2[0]]
                               b = 1
                           else:
                               if cun[0]==0:
                                   bi = [self.T_gh[cun[1]], self.T_gh[s_bh1[-1]]]
                                   zuida_s = bi.index(max(bi))
                                   z_time = bi[zuida_s]
                                   Gui_w[cun[1]][1] = Gui_w[cun[1]][1] + 1
                                   self.T_gh[cun[1]] = z_time + self.shiijan_3([2, Gui_w[cun[1]][1]],
                                                                               [2, Gui_w[cun[1]][1] + 1])
                                   if cun[1]==0:
                                        self.t1=self.t1+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                                   elif cun[1]==1:
                                        self.t2=self.t2+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                                   elif cun[1]==2:
                                        self.t3=self.t3+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                                   self.T_gh[s_bh1[-1]] = z_time
                                   keyong_G = [s_bh1[-1]]
                                   b = 2
                               else:
                                   bi=[self.T_gh[cun[0]],self.T_gh[s_bh2[0]]]
                                   zuida_s = bi.index(max(bi))
                                   z_time=bi[zuida_s]
                                   Gui_w[cun[0]][1] = Gui_w[cun[0]][1] - 1
                                   self.T_gh[cun[0]] = z_time + self.shiijan_3([2, Gui_w[cun[0]][1]],
                                                                                          [2, Gui_w[cun[0]][1] - 1])
                                   if cun[0]==0:
                                        self.t1=self.t1+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                   elif cun[0]==1:
                                        self.t2=self.t2+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                   elif cun[0]==2:
                                        self.t3=self.t3+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                   self.T_gh[s_bh2[0]] = z_time
                                   keyong_G = [s_bh2[0]]
                                   b = 2
                       else:
                           for te in range(self.Gui_num):
                               if Gui_w[cun[0]][1] - 2 == Gui_w[te][1]:
                                   w_n_2=te
                               else:
                                   continue
                           if w_n_2>0:
                               bi=[self.T_gh[cun[1]],self.T_gh[s_bh1[-1]]]
                               zuida_s = bi.index(max(bi))
                               z_time=bi[zuida_s]
                               Gui_w[cun[1]][1] = Gui_w[cun[1]][1] + 1
                               self.T_gh[cun[1]] = z_time + self.shiijan_3([2, Gui_w[cun[1]][1]],
                                                                                      [2, Gui_w[cun[1]][1] + 1])
                               if cun[1]==0:
                                    self.t1=self.t1+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                               elif cun[1]==1:
                                    self.t2=self.t2+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                               elif cun[1]==2:
                                    self.t3=self.t3+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                               self.T_gh[s_bh1[-1]] = z_time
                               keyong_G = [s_bh1[-1]]
                               b = 2
                           else:
                               bi=[self.T_gh[cun[0]],self.T_gh[s_bh2[0]]]
                               zuida_s = bi.index(max(bi))
                               z_time=bi[zuida_s]
                               Gui_w[cun[0]][1] = Gui_w[cun[0]][1] - 1
                               self.T_gh[cun[0]] = z_time + self.shiijan_3([2, Gui_w[cun[0]][1]],
                                                                                      [2, Gui_w[cun[0]][1] - 1])
                               if cun[0]==0:
                                    self.t1=self.t1+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                               elif cun[0]==1:
                                    self.t2=self.t2+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                               elif cun[0]==2:
                                    self.t3=self.t3+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                               self.T_gh[s_bh2[0]] = z_time
                               keyong_G = [s_bh2[0]]
                               b = 2
                   elif len(cun)==1:
                       keyong_G = cun
                   elif cun==[]:
                       keyong_G = [s_bh1[-1], s_bh2[0]]
                   else:
                       print("aaaa")
               elif s_bh1==[] and len(s_bh2):
                    keyong_G = [s_bh2[0]]
               elif len(s_bh1) and s_bh2==[]:
                    keyong_G = [s_bh1[-1]]
               else:
                   print("飞了")
               for j_ in range(len(keyong_G)):
                   t_2 = self.T_gh[keyong_G[j_]] + self.shiijan_3(Gui_w[keyong_G[j_]],self.n_2[a])
                   T_11.append(t_2)
               etiig = T_11
               for c in range(len(keyong_G)):
                    if otiv <= etiig[c]:
                        s_3.append(keyong_G[c])
                        T_3.append(etiig[c]-otiv)
                    else:
                        continue
               if len(s_3):
                   T_3min=T_3.index(min(T_3))
                   self.Gui_J[s_3[T_3min]][i] = a
                   self.T_ah[T_1min]= etiig[T_3min]
                   self.T_gh[s_3[T_3min]] = etiig[T_3min]+self.sigma
                   self.Gui_t[s_3[T_3min]][i] = self.T_gh[s_3[T_3min]]
                   self.F_1.append(max(self.T_ah[T_1min],self.T_gh[s_3[T_3min]],self.TT_1,self.TT_2,self.TT_3))
                   if b==0:
                       self.Fit_E=self.Fit_E+self.E_fitness(self.sigma, self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a]), 0,
                                                            self.shiijan_2(self.n_1[a],self.n_2[a]), self.shiijan_1(self.n_1[a],
                                                                                                                   Agv_w[T_1min]), T_3[T_3min]+cadeng)
                       if s_3[T_3min]==0:
                           self.t1=self.t1+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                       elif s_3[T_3min]==1:
                           self.t2=self.t2+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                       elif s_3[T_3min]==2:
                           self.t3=self.t3+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                   elif b == 1:
                       self.Fit_E = self.Fit_E + self.E_fitness(self.sigma,
                                                                self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])+self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                                                +self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1]), 0,
                                                                self.shiijan_2(self.n_1[a], self.n_2[a]),
                                                                self.shiijan_1(self.n_1[a],
                                                                               Agv_w[T_1min]), T_3[T_3min]+cadeng)
                       if s_3[T_3min]==0:
                           self.t1=self.t1+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                       elif s_3[T_3min]==1:
                           self.t2=self.t2+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                       elif s_3[T_3min]==2:
                           self.t3=self.t3+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                   elif b == 2:
                       self.Fit_E = self.Fit_E + self.E_fitness(self.sigma,
                                                                self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])+self.shiijan_3([2, Gui_w[cun[0]][1]],[2, Gui_w[cun[0]][1] - 1]),
                                                                0,self.shiijan_2(self.n_1[a], self.n_2[a]),self.shiijan_1(self.n_1[a],
                                                                               Agv_w[T_1min]), T_3[T_3min]+cadeng)
                       if s_3[T_3min]==0:
                           self.t1=self.t1+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                       elif s_3[T_3min]==1:
                           self.t2=self.t2+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                       elif s_3[T_3min]==2:
                           self.t3=self.t3+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                   Agv_w[T_1min] = self.n_2[a]
                   Gui_w[s_3[T_3min]] = self.n_2[a]
               else:
                   for l in range(len(keyong_G)):
                       s_2.append(keyong_G[l])
                       T_2_.append(otiv -etiig[l])
                   T_2min= T_2_.index(min(T_2_))
                   self.Gui_J[s_2[T_2min]][i] = a
                   self.T_ah[T_1min] = otiv
                   self.T_gh[s_2[T_2min]] = otiv + self.sigma
                   self.Gui_t[s_2[T_2min]][i]=self.T_gh[s_2[T_2min]]
                   self.F_1.append(max(self.T_ah[T_1min],self.T_gh[s_2[T_2min]],self.TT_1,self.TT_2,self.TT_3))
                   if b==0:
                       self.Fit_E=self.Fit_E+self.E_fitness(self.sigma, self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a]), T_2_[T_2min]
                                                            , self.shiijan_2(self.n_1[a],self.n_2[a]), self.shiijan_1(self.n_1[a],
                                                                                                                      Agv_w[T_1min]), 0+cadeng)
                       if s_2[T_2min]==0:
                           self.t1=self.t1+self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])
                       elif s_2[T_2min]==1:
                           self.t2=self.t2+self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])
                       elif s_2[T_2min]==2:
                           self.t3=self.t3+self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])
                   elif b==1:
                       self.Fit_E=self.Fit_E+self.E_fitness(self.sigma, self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])+self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])+
                                                            self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1]), T_2_[T_2min]
                                                            , self.shiijan_2(self.n_1[a],self.n_2[a]), self.shiijan_1(self.n_1[a],
                                                                                                                      Agv_w[T_1min]), 0+cadeng)
                       if s_2[T_2min]==0:
                           self.t1=self.t1+self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])
                       elif s_2[T_2min]==1:
                           self.t2=self.t2+self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])
                       elif s_2[T_2min]==2:
                           self.t3=self.t3+self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])
                   elif b==2:
                       self.Fit_E = self.Fit_E + self.E_fitness(self.sigma,
                                                                self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])+self.shiijan_3([2, Gui_w[cun[0]][1]],[2, Gui_w[cun[0]][1] - 1]),
                                                                T_2_[T_2min]
                                                                , self.shiijan_2(self.n_1[a], self.n_2[a]),
                                                                self.shiijan_1(self.n_1[a],
                                                                               Agv_w[T_1min]), 0+cadeng)
                       if s_2[T_2min]==0:
                           self.t1=self.t1+self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])
                       elif s_2[T_2min]==1:
                           self.t2=self.t2+self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])
                       elif s_2[T_2min]==2:
                           self.t3=self.t3+self.shiijan_3(Gui_w[s_2[T_2min]], self.n_2[a])
                   Agv_w[T_1min] = self.n_2[a]
                   Gui_w[s_2[T_2min]] = self.n_2[a]



            else:
                for y in range (self.Agv_num):
                    t_1 = self.T_ah[y] + self.shiijan_1(Agv_w[y], self.n_2[a])
                    self.T_1[y] = t_1
                for i__ in range(self.Gui_num):
                    if Gui_w[i__][1]<=self.n_2[a][1]:
                        s_bh1.append(i__)
                    else:
                        s_bh2.append(i__)
                if len(s_bh1) and len(s_bh2):
                    k=[s_bh1[-1],s_bh2[0]]
                    for k_i in k:
                        if Gui_w[k_i][1]+1==self.n_2[a][1]:
                            cun.append(k_i)
                        elif Gui_w[k_i][1]-1==self.n_2[a][1]:
                            cun.append(k_i)
                        else:
                            continue
                    if len(cun)==2:
                        if Gui_w[cun[0]][1]-2 != 0:
                            for te in range(self.Gui_num):
                                if Gui_w[cun[0]][1] - 2 == Gui_w[te][1]:
                                    w_n_2=te
                                else:
                                    continue
                            if w_n_2>0:
                                zuida_s = self.T_gh.index(max(self.T_gh))
                                z_time=self.T_gh[zuida_s]
                                Gui_w[cun[0]][1] = Gui_w[cun[0]][1] - 1
                                Gui_w[w_n_2][1] = Gui_w[w_n_2][1] - 1
                                self.T_gh[cun[0]] = z_time + self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                if cun[0]==0:
                                    self.t1=self.t1+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                elif cun[0]==1:
                                    self.t2=self.t2+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                elif cun[0]==2:
                                    self.t3=self.t3+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                self.T_gh[w_n_2] =z_time+self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1])
                                if w_n_2==0:
                                    self.t1=self.t1+ self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1])
                                elif w_n_2==1:
                                    self.t2=self.t2+ self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1])
                                elif w_n_2==2:
                                    self.t3=self.t3+ self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1])
                                self.T_gh[s_bh2[0]]=z_time
                                keyong_G = [s_bh2[0]]
                                b = 1
                            else:
                                if cun[0]==0:
                                    bi = [self.T_gh[cun[1]], self.T_gh[s_bh1[-1]]]
                                    zuida_s = bi.index(max(bi))
                                    z_time = bi[zuida_s]
                                    Gui_w[cun[1]][1] = Gui_w[cun[1]][1] + 1
                                    self.T_gh[cun[1]] = z_time + self.shiijan_3([2, Gui_w[cun[1]][1]],
                                                                                [2, Gui_w[cun[1]][1] + 1])
                                    if cun[1]==0:
                                        self.t1=self.t1+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                                    elif cun[1]==1:
                                        self.t2=self.t2+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                                    elif cun[1]==2:
                                        self.t3=self.t3+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                                    self.T_gh[s_bh1[-1]] = z_time
                                    keyong_G = [s_bh1[-1]]
                                    b = 2
                                else:
                                    bi=[self.T_gh[cun[0]],self.T_gh[s_bh2[0]]]
                                    zuida_s = bi.index(max(bi))
                                    z_time=bi[zuida_s]
                                    Gui_w[cun[0]][1] = Gui_w[cun[0]][1] - 1
                                    self.T_gh[cun[0]] = z_time + self.shiijan_3([2, Gui_w[cun[0]][1]],
                                                                                            [2, Gui_w[cun[0]][1] - 1])
                                    if cun[0]==0:
                                        self.t1=self.t1+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                    elif cun[0]==1:
                                        self.t2=self.t2+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                    elif cun[0]==2:
                                        self.t3=self.t3+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                    self.T_gh[s_bh2[0]] = z_time
                                    keyong_G = [s_bh2[0]]
                                    b = 2
                        else:
                            for te in range(self.Gui_num):
                                if Gui_w[cun[0]][1] - 2 == Gui_w[te][1]:
                                    w_n_2=te
                                else:
                                    continue
                            if w_n_2>0:
                                bi=[self.T_gh[cun[1]],self.T_gh[s_bh1[-1]]]
                                zuida_s = bi.index(max(bi))
                                z_time=bi[zuida_s]
                                Gui_w[cun[1]][1] = Gui_w[cun[1]][1] + 1
                                self.T_gh[cun[1]] = z_time + self.shiijan_3([2, Gui_w[cun[1]][1]],
                                                                                        [2, Gui_w[cun[1]][1] + 1])
                                if cun[1]==0:
                                    self.t1=self.t1+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                                elif cun[1]==1:
                                    self.t2=self.t2+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                                elif cun[1]==2:
                                    self.t3=self.t3+ self.shiijan_3([2,Gui_w[cun[1]][1]],[2,Gui_w[cun[1]][1] + 1])
                                self.T_gh[s_bh1[-1]] = z_time
                                keyong_G = [s_bh1[-1]]
                                b = 2
                            else:
                                bi=[self.T_gh[cun[0]],self.T_gh[s_bh2[0]]]
                                zuida_s = bi.index(max(bi))
                                z_time=bi[zuida_s]
                                Gui_w[cun[0]][1] = Gui_w[cun[0]][1] - 1
                                self.T_gh[cun[0]] = z_time + self.shiijan_3([2, Gui_w[cun[0]][1]],
                                                                                        [2, Gui_w[cun[0]][1] - 1])
                                if cun[0]==0:
                                    self.t1=self.t1+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                elif cun[0]==1:
                                    self.t2=self.t2+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                elif cun[0]==2:
                                    self.t3=self.t3+ self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])
                                self.T_gh[s_bh2[0]] = z_time
                                keyong_G = [s_bh2[0]]
                                b = 2
                    elif len(cun)==1:
                        keyong_G = cun
                    elif cun==[]:
                        keyong_G = [s_bh1[-1], s_bh2[0]]
                    else:
                        print("aaaa")
                elif s_bh1==[] and len(s_bh2):
                    keyong_G = [s_bh2[0]]
                elif len(s_bh1) and s_bh2==[]:
                    keyong_G = [s_bh1[-1]]
                else:
                    print("飞了")
                for j_ in range(len(keyong_G)):
                    t_2 = self.T_gh[keyong_G[j_]] + self.shiijan_3(Gui_w[keyong_G[j_]], self.n_2[a]) + self.sigma
                    T_11.append(t_2)
                etiig = T_11
                for c in range(len(keyong_G)):
                    for c_ in range(self.Agv_num):
                        if self.T_1[c_] <= etiig[c]:
                            s_3.append(keyong_G[c])
                            s_4.append(c_)
                            T_3.append(etiig[c] - self.T_1[c_])
                        else:
                            continue
                if len(s_3):
                    T_3min = T_3.index(min(T_3))
                    suoyin = 0
                    for i2 in range(len(keyong_G)):
                        if keyong_G[i2]==s_3[T_3min]:
                            suoyin=i2
                        else:
                            continue
                    self.T_ah[s_4[T_3min]] = etiig[suoyin]+self.shiijan_2(self.n_2[a], self.n_1[a])
                    if self.n_1[a][1] == 21:
                        if self.T_ah[s_4[T_3min]] <= (self.TT_1 + self.TTZ):
                            for ii in range(4):
                                tc = self.T_ah[s_4[T_3min]] - self.T_lb1[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblza.append(ii)
                            if cqblza == []:
                                k2 = cqjs.index(max(cqjs))

                                if (self.TT_1 + self.TTZ) >= self.T_lb1[k2]:
                                    self.TT_1 = self.TT_1 + 2*self.TTZ
                                    cadeng = self.T_lb1[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb1[k2]
                                    self.T_lb1[k2] = self.TT_1
                                else:
                                    self.TT_1 = self.T_lb1[k2]+self.TTZ
                                    cadeng = self.T_lb1[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb1[k2]
                            else:
                                k = cqblza[0]

                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_1 = self.TT_1 + 2*self.TTZ
                                self.T_lb1[k] = self.TT_1

                        else:
                            for ii in range(4):
                                tc = self.TT_1 + self.TTZ - self.T_lb1[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblzg.append(ii)
                            if cqblzg == []:
                                k2 = cqjs.index(max(cqjs))
                                if self.T_lb1[k2] >= self.T_ah[s_4[T_3min]]:
                                    self.TT_1 = self.T_lb1[k2]+self.TTZ
                                    cadeng = self.T_lb1[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb1[k2]

                                else:
                                    self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                    self.TT_1 = self.T_ah[s_4[T_3min]]+self.TTZ
                                    self.T_lb1[k2] = self.T_ah[s_4[T_3min]]

                            else:
                                k1 = cqblzg[0]

                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_1 = self.T_ah[s_4[T_3min]]+self.TTZ
                                self.T_lb1[k1] = self.T_ah[s_4[T_3min]]

                    if self.n_1[a][1] == 25:
                        if self.T_ah[s_4[T_3min]] <= (self.TT_2 + self.TTZ):
                            for ii in range(4):
                                tc = self.T_ah[s_4[T_3min]] - self.T_lb2[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblza.append(ii)
                            if cqblza == []:
                                k2 = cqjs.index(max(cqjs))

                                if (self.TT_2 + self.TTZ) >= self.T_lb2[k2]:
                                    self.TT_2 = self.TT_2 + 2*self.TTZ
                                    cadeng = self.T_lb2[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb2[k2]
                                    self.T_lb2[k2] = self.TT_2
                                else:
                                    self.TT_2 = self.T_lb2[k2]+self.TTZ
                                    cadeng = self.T_lb2[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb2[k2]

                            else:
                                k = cqblza[0]

                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_2 = self.TT_2 + 2*self.TTZ
                                self.T_lb2[k] = self.TT_2

                        else:
                            for ii in range(4):
                                tc = self.TT_2 + self.TTZ - self.T_lb2[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblzg.append(ii)
                            if cqblzg == []:
                                k2 = cqjs.index(max(cqjs))
                                if self.T_lb2[k2] >= self.T_ah[s_4[T_3min]]:
                                    self.TT_2 = self.T_lb2[k2]+self.TTZ
                                    cadeng = self.T_lb2[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb2[k2]
                                else:
                                    self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                    self.TT_2 = self.T_ah[s_4[T_3min]]+self.TTZ
                                    self.T_lb2[k2] = self.T_ah[s_4[T_3min]]

                            else:
                                k1 = cqblzg[0]
                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_2 = self.T_ah[s_4[T_3min]]+self.TTZ
                                self.T_lb2[k1] = self.T_ah[s_4[T_3min]]

                    if self.n_1[a][1] == 29:
                        if self.T_ah[s_4[T_3min]] <= (self.TT_3 + self.TTZ):
                            for ii in range(4):
                                tc = self.T_ah[s_4[T_3min]] - self.T_lb3[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblza.append(ii)
                            if cqblza == []:
                                k2 = cqjs.index(max(cqjs))
                                if (self.TT_3 + self.TTZ) >= self.T_lb3[k2]:
                                    self.TT_3 = self.TT_3 + 2*self.TTZ
                                    cadeng = self.T_lb3[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb3[k2]
                                    self.T_lb3[k2] = self.TT_3
                                else:
                                    self.TT_3 = self.T_lb3[k2]+self.TTZ
                                    cadeng = self.T_lb3[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb3[k2]
                            else:
                                k = cqblza[0]
                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_3 = self.TT_3 + 2*self.TTZ
                                self.T_lb3[k] = self.TT_3

                        else:
                            for ii in range(4):
                                tc = self.TT_3 + self.TTZ - self.T_lb3[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblzg.append(ii)
                            if cqblzg == []:
                                k2 = cqjs.index(max(cqjs))
                                if self.T_lb3[k2] >= self.T_ah[s_4[T_3min]]:
                                    self.TT_3 = self.T_lb3[k2]+self.TTZ
                                    cadeng = self.T_lb3[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb3[k2]
                                else:
                                    self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                    self.TT_3 = self.T_ah[s_4[T_3min]]+self.TTZ
                                    self.T_lb3[k2] = self.T_ah[s_4[T_3min]]

                            else:
                                k1 = cqblzg[0]

                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_3 = self.T_ah[s_4[T_3min]]+self.TTZ
                                self.T_lb3[k1] = self.T_ah[s_4[T_3min]]

                    self.T_ah[s_4[T_3min]] = self.T_ah[s_4[T_3min]]+cadeng
                    self.T_gh[s_3[T_3min]] = etiig[suoyin]
                    self.Gui_J[s_3[T_3min]][i]=a
                    self.Agv_J[s_4[T_3min]][i]=a
                    self.F_1.append(max(self.T_ah[s_4[T_3min]], self.T_gh[s_3[T_3min]],self.TT_1,self.TT_2,self.TT_3))
                    if b==0:
                        self.Fit_E = self.Fit_E + self.E_fitness(self.sigma, self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a]),
                                                                 0,
                                                                 self.shiijan_2(self.n_2[a], self.n_1[a]),
                                                                 self.shiijan_1(Agv_w[s_4[T_3min]], self.n_2[a]), T_3[T_3min]+cadeng)
                        if s_3[T_3min]==0:
                            self.t1=self.t1+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==1:
                            self.t2=self.t2+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==2:
                            self.t3=self.t3+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                    elif b==1:
                        self.Fit_E = self.Fit_E + self.E_fitness(self.sigma, self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])+self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])+
                                                            self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1]),
                                                                 0,
                                                                 self.shiijan_2(self.n_2[a], self.n_1[a]),
                                                                 self.shiijan_1(Agv_w[s_4[T_3min]], self.n_2[a]), T_3[T_3min]+cadeng)
                        if s_3[T_3min]==0:
                            self.t1=self.t1+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==1:
                            self.t2=self.t2+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==2:
                            self.t3=self.t3+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                    elif b==2:
                        self.Fit_E = self.Fit_E + self.E_fitness(self.sigma, self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])+self.shiijan_3([2, Gui_w[cun[0]][1]],[2, Gui_w[cun[0]][1] - 1]),
                                                                 0,
                                                                 self.shiijan_2(self.n_2[a], self.n_1[a]),
                                                                 self.shiijan_1(Agv_w[s_4[T_3min]], self.n_2[a]), T_3[T_3min]+cadeng)
                        if s_3[T_3min]==0:
                            self.t1=self.t1+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==1:
                            self.t2=self.t2+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==2:
                            self.t3=self.t3+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                    Agv_w[s_4[T_3min]] = self.n_1[a]
                    Gui_w[s_3[T_3min]] = self.n_2[a]
                else:
                    for c1 in range(len(keyong_G)):
                        for c1_ in range(self.Agv_num):
                            s_3.append(keyong_G[c1])
                            s_4.append(c1_)
                            T_3.append(self.T_1[c1_]-etiig[c1])
                    T_3min = T_3.index(min(T_3))
                    self.T_ah[s_4[T_3min]] = self.T_1[s_4[T_3min]] + self.shiijan_2(self.n_1[a],self.n_2[a])
                    if self.n_1[a][1] == 21:
                        if self.T_ah[s_4[T_3min]] <= (self.TT_1 + self.TTZ):
                            for ii in range(4):
                                tc = self.T_ah[s_4[T_3min]] - self.T_lb1[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblza.append(ii)
                            if cqblza == []:
                                k2 = cqjs.index(max(cqjs))
                                if (self.TT_1 + self.TTZ) >= self.T_lb1[k2]:
                                    self.TT_1 = self.TT_1 + 2*self.TTZ
                                    cadeng = self.T_lb1[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb1[k2]
                                    self.T_lb1[k2] = self.TT_1
                                else:
                                    self.TT_1 = self.T_lb1[k2]+self.TTZ
                                    cadeng = self.T_lb1[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb1[k2]

                            else:
                                k = cqblza[0]

                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_1 = self.TT_1 + 2*self.TTZ
                                self.T_lb1[k] = self.TT_1

                        else:
                            for ii in range(4):
                                tc = self.TT_1 + self.TTZ - self.T_lb1[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblzg.append(ii)
                            if cqblzg == []:
                                k2 = cqjs.index(max(cqjs))
                                if self.T_lb1[k2] >= self.T_ah[s_4[T_3min]]:
                                    self.TT_1 = self.T_lb1[k2]+self.TTZ
                                    cadeng = self.T_lb1[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb1[k2]
                                else:
                                    self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                    self.TT_1 = self.T_ah[s_4[T_3min]]+self.TTZ
                                    self.T_lb1[k2] = self.T_ah[s_4[T_3min]]

                            else:
                                k1 = cqblzg[0]

                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_1 = self.T_ah[s_4[T_3min]]+self.TTZ
                                self.T_lb1[k1] = self.T_ah[s_4[T_3min]]

                    if self.n_1[a][1] == 25:
                        if self.T_ah[s_4[T_3min]] <= (self.TT_2 + self.TTZ):
                            for ii in range(4):
                                tc = self.T_ah[s_4[T_3min]] - self.T_lb2[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblza.append(ii)
                            if cqblza == []:
                                k2 = cqjs.index(max(cqjs))
                                if (self.TT_2 + self.TTZ) >= self.T_lb2[k2]:
                                    self.TT_2 = self.TT_2 + 2*self.TTZ
                                    cadeng = self.T_lb2[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb2[k2]
                                    self.T_lb2[k2] = self.TT_2
                                else:
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb2[k2]
                                    self.TT_2 = self.T_lb2[k2]+self.TTZ
                                    cadeng = self.T_lb2[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb2[k2]

                            else:
                                k = cqblza[0]

                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_2 = self.TT_2 + 2*self.TTZ
                                self.T_lb2[k] = self.TT_2

                        else:
                            for ii in range(4):
                                tc = self.TT_2 + self.TTZ - self.T_lb2[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblzg.append(ii)
                            if cqblzg == []:
                                k2 = cqjs.index(max(cqjs))
                                if self.T_lb2[k2] >= self.T_ah[s_4[T_3min]]:
                                    self.TT_2 = self.T_lb2[k2]+self.TTZ
                                    cadeng = self.T_lb2[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb2[k2]

                                else:
                                    self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                    self.TT_2 = self.T_ah[s_4[T_3min]]+self.TTZ
                                    self.T_lb2[k2] = self.T_ah[s_4[T_3min]]

                            else:
                                k1 = cqblzg[0]

                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_2 = self.T_ah[s_4[T_3min]]+self.TTZ
                                self.T_lb2[k1] = self.T_ah[s_4[T_3min]]

                    if self.n_1[a][1] == 29:
                        if self.T_ah[s_4[T_3min]] <= (self.TT_3 + self.TTZ):
                            for ii in range(4):
                                tc = self.T_ah[s_4[T_3min]] - self.T_lb3[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblza.append(ii)
                            if cqblza == []:
                                k2 = cqjs.index(max(cqjs))
                                if (self.TT_3 + self.TTZ) >= self.T_lb3[k2]:
                                    self.TT_3 = self.TT_3 + 2*self.TTZ
                                    cadeng = self.T_lb3[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb3[k2]
                                    self.T_lb3[k2] = self.TT_3
                                else:
                                    self.TT_3 = self.T_lb3[k2]+self.TTZ
                                    cadeng = self.T_lb3[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb3[k2]
                            else:
                                k = cqblza[0]
                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_3 = self.TT_3 + 2*self.TTZ
                                self.T_lb3[k] = self.TT_3

                        else:
                            for ii in range(4):
                                tc = self.TT_3 + self.TTZ - self.T_lb3[ii]
                                cqjs.append(tc)
                                if tc >= 0:
                                    cqblzg.append(ii)
                            if cqblzg == []:
                                k2 = cqjs.index(max(cqjs))
                                if self.T_lb3[k2] >= self.T_ah[s_4[T_3min]]:
                                    self.TT_3 = self.T_lb3[k2]+self.TTZ
                                    cadeng = self.T_lb3[k2] - self.T_ah[s_4[T_3min]]
                                    self.Agv_t[s_4[T_3min]][i] = self.T_lb3[k2]
                                else:
                                    self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                    self.TT_3 = self.T_ah[s_4[T_3min]]+self.TTZ
                                    self.T_lb3[k2] = self.T_ah[s_4[T_3min]]

                            else:
                                k1 = cqblzg[0]

                                self.Agv_t[s_4[T_3min]][i] = self.T_ah[s_4[T_3min]]
                                self.TT_3 = self.T_ah[s_4[T_3min]]+self.TTZ
                                self.T_lb3[k1] = self.T_ah[s_4[T_3min]]

                    self.T_ah[s_4[T_3min]] = self.T_ah[s_4[T_3min]]+cadeng
                    self.T_gh[s_3[T_3min]] = self.T_1[s_4[T_3min]]
                    self.Gui_J[s_3[T_3min]][i]=a
                    self.Agv_J[s_4[T_3min]][i]=a
                    zc = max(self.T_ah[s_4[T_3min]], self.T_gh[s_3[T_3min]],self.TT_1,self.TT_2,self.TT_3)
                    self.F_1.append(zc)
                    if b==0:
                        self.Fit_E = self.Fit_E + self.E_fitness(self.sigma,self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a]),T_3[T_3min], self.shiijan_2(self.n_1[a], self.n_2[a]),self.shiijan_1(Agv_w[s_4[T_3min]], self.n_2[a]), 0+cadeng)
                        if s_3[T_3min]==0:
                            self.t1=self.t1+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==1:
                            self.t2=self.t2+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==2:
                            self.t3=self.t3+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                    elif b==1:
                        self.Fit_E = self.Fit_E + self.E_fitness(self.sigma,
                                                                 self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])+self.shiijan_3([2,Gui_w[cun[0]][1]],[2,Gui_w[cun[0]][1] - 1])+
                                                            self.shiijan_3([2,Gui_w[w_n_2][1]],[2,Gui_w[w_n_2][1] - 1]),
                                                                 T_3[T_3min]
                                                                 , self.shiijan_2(self.n_1[a], self.n_2[a]),
                                                                 self.shiijan_1(Agv_w[s_4[T_3min]], self.n_2[a]), 0+cadeng)
                        if s_3[T_3min]==0:
                            self.t1=self.t1+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==1:
                            self.t2=self.t2+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==2:
                            self.t3=self.t3+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                    elif b==2:
                        self.Fit_E = self.Fit_E + self.E_fitness(self.sigma,
                                                                 self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])+self.shiijan_3([2, Gui_w[cun[0]][1]],[2, Gui_w[cun[0]][1] - 1]),
                                                                 T_3[T_3min]
                                                                 , self.shiijan_2(self.n_1[a], self.n_2[a]),
                                                                 self.shiijan_1(Agv_w[s_4[T_3min]], self.n_2[a]), 0+cadeng)
                        if s_3[T_3min]==0:
                            self.t1=self.t1+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==1:
                            self.t2=self.t2+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                        elif s_3[T_3min]==2:
                            self.t3=self.t3+self.shiijan_3(Gui_w[s_3[T_3min]], self.n_2[a])
                    Agv_w[s_4[T_3min]] = self.n_1[a]
                    Gui_w[s_3[T_3min]]= self.n_2[a]
            st=[x[1] for x in Gui_w]  
            self.Gui_wz.append(st)
            tt=copy.deepcopy(self.T_gh)
            self.Gui_tz.append(tt)







import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.patches as patches
import matplotlib
#matplotlib.use("nbagg")
import matplotlib.animation as anm
import time
import sys
from numpy import linalg as LA
from scipy.optimize import fsolve

class World:

    def __init__(self,time_span,time_interval,debug=False):
        self.robot = 0
        self.objects = []
        self.betweens = []
        self.debug = debug
        self.time_interval = time_interval
        self.time_span = time_span

    def robot_append(self,robot):
        self.robot= robot

    def obj_append(self,obj):
        self.objects.append(obj)

    def bet_append(self,bet):
        self.betweens.append(bet)

    def one_step(self,i,elems,ax):
        while elems:
            elems.pop().remove()
        #elems.append(ax.text(-4.4,4.5,"t = "+str(i),fontsize=10))
        self.robot.draw(ax,elems)
        if hasattr(self.robot,"one_step"):
            self.robot.one_step(self.time_interval)
        for obj in self.objects:
            obj.draw(ax,elems)
            if hasattr(obj,"one_step"):
                obj.one_step(self.time_interval)
        for bet in self.betweens:
            bet.draw(ax,elems)
            if hasattr(bet,"one_step"):
                bet.one_step(self.time_interval)
        pot = LJ()
        if pot.goal_x-robot.pose[0] < 0.12 and pot.goal_y-robot.pose[1] < 0.12:
            sys.exit()
        

    def draw(self):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_xlim(-1,10)
        ax.set_ylim(5,-1)
        ax.set_xlabel("x",fontsize=20)
        ax.set_ylabel("y",fontsize=20)
        ax.scatter(2,0,marker="*",s=100)
        ax.scatter(9.8,4.8,marker="*",s=100)

        elems = []

        if self.debug:
            for i in range(1000):
                self.one_step(i, elems, ax)
        else:

            self.ani = anm.FuncAnimation(fig,self.one_step,fargs=(elems,ax),frames=int(self.time_span/self.time_interval)+1,interval=int(self.time_interval*1000),repeat=False)
            self.ani.save("seiteki.gif",writer="imagemagick")
            plt.show()

class LJ:
    p = 2
    q = 1
    r_a  = 2.0 #要検討，人間が人混みを歩くときにどの辺りまで把握しているか．歩行者密度を計測するための半径
    r_c = 4
    r_ro = 0.3
    r_hu = 0.3 #歩行者の半径
    sigma_wn_2 = 1/5
    w = 7.0*1e-3
    s = 0
    e_lj = 2
    lj_jouken = False
    goal_jouken = False
    goryu_jouken = False
    
    obsts = []
    betweens = []
    robot = 0
    vector_goal = np.array([10,0])
    delt = 1/10
    speed = 1.3
    goal_x, goal_y = 10,5
    weight_goal = 40

    def obst_append(self,obst):
        self.obsts.append(obst)

    def bet_append(self,bet):
        self.betweens.append(bet)

    def obst_xy(self):
        self.obst_x = []
        self.obst_y = []
        self.obst_w = []
        self.obst = []
        for obs in self.obsts:
            self.obst_x.append(obs.pose[0])
            self.obst_y.append(obs.pose[1])
            self.obst_w.append(obs.pose[2])
            self.obst.append(obs.pose[0:2])

    def bet_xy(self):
        self.bet_x = []
        self.bet_y = []
        for bet in self.betweens:
            self.bet_x.append(bet.pose[0])
            self.bet_y.append(bet.pose[1])

    # robotと障害物の距離を計算
    def robot_obst_dist(self,x,y):
        obst_dist = []
        num = len(self.obsts)

        for i in range(num):
            obst_dist.append(np.sqrt((x-self.obst_x[i])**2+(y-self.obst_y[i])**2))

        if LJ.goryu_jouken:
            for i in range(num):
                if obst_dist[i] > self.r_c or self.obst_x[i] < 0 or self.obst_x[i] < x:
                    obst_dist[i] = None
        else:
            for i in range(num):
                if obst_dist[i] > self.r_c or self.obst_x[i] < 0:
                    obst_dist[i] = None

        return obst_dist,num

    #歩行者とrobotの距離を計算
    def robot_bet_dist(self,x,y):
        bet_dist= []
        kaisu = len(self.betweens)

        for i in range(kaisu):
            bet_dist.append(np.sqrt((x-self.bet_x[i])**2+(y-self.bet_y[i])**2))

        if LJ.goryu_jouken:
            for i in range(kaisu):
                if bet_dist[i] > self.r_c or self.bet_x[i] < 0 or self.bet_x[i] < x:
                    bet_dist[i] = None
        else:
            for i in range(kaisu):
                if bet_dist[i] > self.r_c or self.bet_x[i] < 0:
                    bet_dist[i] = None

        return bet_dist,kaisu

    #歩行者群の進行方向及び目的地方向との角度差
    def angle_obst(self):
        pose_bf = []
        pose_af = []
        degree = np.zeros(len(self.obsts))
        e = np.zeros((len(self.obsts),2)) #単位ベクトルを入れる

        for obs in self.obsts:
            pose_bf.append(obs.pose[0:2])
            obs.step(self.delt)
            pose_af.append(obs.pose_af[0:2])
        pose_bf = np.array(pose_bf)
        pose_af = np.array(pose_af)
        vec_obs = pose_af - pose_bf
        vec_obs = np.array(vec_obs)

        for index in range(len(self.obsts)):
            i = np.inner(self.vector_goal,vec_obs[index])
            n = LA.norm(self.vector_goal)*LA.norm(vec_obs[index])
            if n==0:
                deg = np.pi
            else:
                c = i/n
                deg = np.arccos(np.clip(c,-1.0,1.0))
            degree[index] = deg

        #進行方向とは逆の単位ベクトルを作成
        for index in range(len(self.obsts)):
            n = LA.norm(vec_obs[index])
            e[index] = vec_obs[index] / n
            e[index] = -e[index]

        return degree, e

    #各歩行者の進行方向及び目的地方向との角度差
    def angle_bet(self):
        pose_bf = []
        pose_af = []
        bet_degree = np.zeros(len(self.betweens))

        for bet in self.betweens:
            pose_bf.append(bet.pose[0:2])
            bet.step(self.delt)
            pose_af.append(bet.pose_af[0:2])
        pose_bf = np.array(pose_bf)
        pose_af = np.array(pose_af)
        vec_obs = pose_af - pose_bf
        vec_obs = np.array(vec_obs)

        for index in range(len(self.betweens)):
            i = np.inner(self.vector_goal,vec_obs[index])
            n = LA.norm(self.vector_goal)*LA.norm(vec_obs[index])
            if n==0:
                deg = np.pi
            else:
                c = i/n
                deg = np.arccos(np.clip(c,-1.0,1.0))
            bet_degree[index] = deg
        return bet_degree

    def f(self,theta):
        sum = 0
        for i in range(-10,11):
            a = np.exp(-(theta+2*np.pi*i)**2/(2*self.sigma_wn_2))
            sum += a

        return (1/(np.sqrt(2*np.pi*self.sigma_wn_2)))*sum

    def weight(self,theta):
        a = self.f(theta)/self.f(0)
        b = (1-self.w)*a+self.w
        return a,b

    def cal_sigma_lj(self,x,y,bet_dist,kaisu):
        betweens = self.betweens
        dist_pedestrian = []
        temp = []
        tmp = 0
        sigma_lj = 0.6 #デフォルトのsigma_lj
        sigma_lj_tmp = 0
        dist_pedestrian_mean = 0

        #歩行者同士の距離を求め，それが指定の値より大きい場合は欠損値とする
        #さらに障害物の距離がNoneの場合は，欠損値とする
        for i in range(kaisu):
            for j in range(kaisu):
                if i == j:
                    pass
                elif bet_dist[i] == None or bet_dist[j] == None:
                    temp.append(np.nan)
                else:
                    tmp = np.sqrt((self.bet_x[i] - self.bet_x[j])**2+(self.bet_y[i] - self.bet_y[j])**2)
                    if tmp > 1.8:
                        temp.append(np.nan)
                    else:
                        temp.append(tmp)
            dist_pedestrian.append(temp)
            temp = []

        #歩行者同士の距離の平均を取って，歩行者間の平均距離を求める
        #歩行者の平均距離を求められなかった場合も考える
        dist_pedestrian_mean = np.nanmean(dist_pedestrian)

        if np.isnan(dist_pedestrian_mean):
            #ロボットと歩行者の最低限の距離離れさす，これは合流に使用
            return sigma_lj, 0.6
        else:
            #平均歩行者距離から，ロボットと歩行者の距離を決めるsigma_ljを求める
            sigma_lj_tmp = (dist_pedestrian_mean - (self.r_ro + self.r_hu)) / 2

            #歩行者流に合流したらsigma_ljを変化させる
            if LJ.goryu_jouken:
                if sigma_lj < sigma_lj_tmp:
                    return sigma_lj, dist_pedestrian_mean
                else:
                    return sigma_lj_tmp, dist_pedestrian_mean
            else:
                return sigma_lj, dist_pedestrian_mean


    def func(self,start,robot_x,robot_y,obst_x,obst_y,width,height,w):
        x, y = start[0], start[1]
        f = np.zeros(2)
        f[0] = y - (robot_y-obst_y)/(robot_x-obst_x)*(x-obst_x) - obst_y
        #角度θ回転した場合
        #楕円角度を入力できるようにする
        f[1] = np.power(((x-obst_x)*np.cos(w*np.pi/180)+(y-obst_y)*np.sin(w*np.pi/180))/max(width,height),2)+np.power((-(x-obst_x)*np.sin(w*np.pi/180)+(y-obst_y)*np.cos(w*np.pi/180))/min(width,height),2)-1
        return f

    def cal_r_gr(self,robot_x,robot_y,num):
        cross = []
        tmp = 0
        
        r_gr = np.zeros(num)
        for i in range(num):
            tmp = fsolve(self.func,[robot_x,robot_y],(robot_x,robot_y,self.obst_x[i],self.obst_y[i],self.obsts[i].width/2,self.obsts[i].height/2,self.obst_w[i]))
            cross.append(tmp)
            r_gr[i] = np.sqrt((cross[i][0]-self.obst_x[i])**2+(cross[i][1]-self.obst_y[i])**2)
            #plt.plot(cross[i][0],cross[i][1],"o")

        return r_gr

    def cal_pot(self,x,y,theta_obst,theta_bet,sigma_lj,r_gr,goryu):
        tmp_pot = 0
        pot_all = 0

        #ロボット位置座標で障害物，歩行者の距離が変化するためcal_potで計算する必要あり
        bet_dist,kaisu = self.robot_bet_dist(x,y)
        obst_dist, num = self.robot_obst_dist(x,y)

        # distの中身に1つでも数字があればLJポテンシャルを発生
        if LJ.lj_jouken:
            # もしロボットのセンシング範囲内にゴールが入ればゴールにポテンシャルを発生
            if LJ.goal_jouken:
                """
                #歩行者に斥力を発生させる
                for i in range(kaisu):
                    if bet_dist[i] == None:
                        tmp_pot = 0
                        pot_all += tmp_pot
                    else:
                        tmp_pot = 1/math.sqrt(pow((x-self.bet_x[i]),2)+pow((y-self.bet_y[i]),2))
                        pot_all += tmp_pot*1.9
                """
                tmp_pot = -1/math.sqrt(pow((x-self.goal_x),2)+pow((y-self.goal_y),2))
                pot_all += tmp_pot * self.weight_goal

            # そうでなければLJポテンシャルを発生
            else:
                #歩行者流に合流できたら，歩行者にだけポテンシャルを発生
                #各歩行者に引力を発生
                if LJ.goryu_jouken:
                    for i in range(kaisu):
                        if bet_dist[i] == None:
                            tmp_pot = 0
                            pot_all += tmp_pot
                        else:
                            a,b = self.weight(theta_bet[i])
                            r = bet_dist[i]
                            tmp_pot = 4*self.e_lj*(b*((sigma_lj+self.s*(1-a))/(r-(self.r_ro+self.r_hu)))**self.p-a*((sigma_lj+self.s*(1-a))/(r-(self.r_ro+self.r_hu)))**self.q)
                            pot_all += tmp_pot
                else:
                    #歩行者群にLJポテンシャルを発生
                    for i in range(num):
                        if obst_dist[i] == None:
                            tmp_pot = 0
                            pot_all += tmp_pot
                        else:
                            a,b = self.weight(theta_obst[i])
                            r = obst_dist[i]
                            tmp_pot = 4*self.e_lj*(b*((sigma_lj+self.s*(1-a))/(r-(self.r_ro+r_gr[i])))**self.p-a*((sigma_lj+self.s*(1-a))/(r-(self.r_ro+r_gr[i])))**self.q)
                            pot_all += tmp_pot

                    #合流地点に引力を生成するプログラム
                    if goryu[0] == None or goryu[1] == None:
                        tmp_pot = 0
                        pot_all += tmp_pot
                    else:
                        goryu_pot = -1/math.sqrt(pow((x-goryu[0]),2)+pow((y-goryu[1]),2))
                        goryu_pot = goryu_pot * self.weight_goal #重みを変更して歩行者流の中央に合流させる，引力が大きくないと歩行者のポテンシャルにつられてしまう
                        pot_all += goryu_pot

        # 全てがNoneであれば，ゴールにポテンシャルを発生させる
        else:
            pot_all = -1/math.sqrt(pow((x-self.goal_x),2)+pow((y-self.goal_y),2))
            pot_all = pot_all * self.weight_goal

        return pot_all

    #条件を切り替える関数
    def condition_change(self,x,y,obst_dist,num,goryu):

        #self.jouken=Trueとしてしまうと，self.joukenというインスタンス変数が作られてクラス変数が隠れてしまう
        #そのため，クラス.クラス変数とするとクラス変数が変更される
        #歩行者群との距離がNoneでなければLJポテンシャルを発生させる
        ##合流した後は，歩行者群ではなく，各歩行者がNoneでなければLJポテンシャルを発生させるにする
        if any(obst_dist):
            LJ.lj_jouken=True
        else:
            LJ.lj_jouken=False

        #歩行者群の中央に合流できたら，中央部分に発生する引力を0にする
        ##合流部分がNone，センシング範囲外の場合も考える
        if goryu[0] == None or goryu[1] == None:
            pass
        elif np.sqrt((x-goryu[0])**2+(y-goryu[1])**2) < 0.1:
            LJ.goryu_jouken = True
            LJ.speed = 1

        else:
            pass

        #ロボットのセンシング範囲内に目的地が入れば，目的地にだけ引力を生成
        if np.sqrt((x-self.goal_x)**2+(y-self.goal_y)**2) < self.r_c:
            LJ.goal_jouken = True
            LJ.speed=1.3


    #合流したら計算しなくていいようにする
    def cal_goryu(self,x,y,theta_obst, e, num, dist_pedestrian_mean):
        goryus = np.zeros((num,2))
        goryu = []
        dist = 0
        tmp = 0

        #合流したら，合流地点を計算するのを止める
        #合流して，その歩行者流が目的地方向とは別の方向に進んだら，歩行者流から脱出して別の歩行者群にLJポテンシャルを生成する
        if LJ.goryu_jouken:
            goryu = [None,None]
            return goryu
        else:
            #各歩行者群について，進行方向とは反対方向に合流地点を生成
            for i in range(num):
                #歩行者群の目的地方向との角度差が0～90°であれば合流地点を生成
                if theta_obst[i] < 90*np.pi/180 and theta_obst[i] >= 0:
                    #進行方向を大きさが1の単位ベクトルとして考え，反対方向に移動したい距離×単位ベクトルにする
                    #歩行者群は進行方向に縦長であると仮定して，楕円の長軸+歩行者間平均距離分移動させる
                    if self.obsts[i].width > self.obsts[i].height:
                        goryus[i] = self.obst[i] + (self.obsts[i].width/2 + 0.6) * e[i]

                        #合流地点が求まれば，ロボットからの距離が一番近い合流地点を選択
                        #歩行者群の角度差で合流地点を生成する，しないを決めても良いかも
                        tmp = np.sqrt((x-goryus[i][0])**2+(y-goryus[i][1])**2)

                        if tmp > self.r_c:
                            pass
                        else:
                            #ロボットと合流地点の距離が一番短い合流地点を出力
                            if len(goryu) == 0:
                                dist = tmp
                                goryu = goryus[i]
                            elif dist > tmp:
                                dist = tmp
                                goryu = goryus[i]
                            else:
                                pass
                    else:
                        goryus[i] = self.obst[i] + (self.obsts[i].height/2 + 0.6) * e[i]
                
                        #合流地点が求まれば，ロボットからの距離が一番近い合流地点を選択
                        #歩行者群の角度差で合流地点を生成する，しないを決めても良いかも
                        tmp = np.sqrt((x-goryus[i][0])**2+(y-goryus[i][1])**2)

                        #ロボットと合流地点の距離が一番短い合流地点を出力
                        if len(goryu) == 0:
                            dist = tmp
                            goryu = goryus[i]
                        elif dist > tmp:
                            dist = tmp
                            goryu = goryus[i]
                        else:
                            pass
                else:
                    pass

            #歩行者群が全て90°以上，合流地点との距離がセンシング範囲外であるときはNoneを返す
            if len(goryu) == 0:
                goryu = [None,None]
                return goryu
            else:
                #plt.plot(goryu[0],goryu[1],"o")
                return goryu

    def cal_route(self,x,y):
        #障害物のxy座標を格納
        self.bet_xy()
        self.obst_xy()

        #角度差を出力
        #歩行者群，各歩行者の目的地方向との角度差を出力して，LJポテンシャルに利用
        theta_obst, e = self.angle_obst()
        theta_bet = self.angle_bet()

        #障害物，歩行者の距離
        #合流したらロボットより後ろの歩行者を認識しなくするため
        obst_dist, num = self.robot_obst_dist(x,y)
        bet_dist, kaisu = self.robot_bet_dist(x,y)

        #sigma_ljを出力
        sigma_lj, dist_pedestrian_mean = self.cal_sigma_lj(x,y,bet_dist,kaisu)

        #合流地点生成プログラム
        #歩行者群の後に引力を発生させる
        #角度差が0°の歩行者群の前•後に引力を発生させて，合流させる．
        goryu = self.cal_goryu(x,y,theta_obst,e,num,dist_pedestrian_mean)
        #goryu = [None,None]

        #条件を切り替える関数を作成
        #下で同じ関数を繰り返すのは，条件を切り替える関数で条件が変わってしまったから
        self.condition_change(x,y,obst_dist,num,goryu)

        #障害物，歩行者の距離
        #合流したらロボットより後ろの歩行者を認識しなくするため
        obst_dist, num = self.robot_obst_dist(x,y)
        bet_dist, kaisu = self.robot_bet_dist(x,y)

        #sigma_ljを出力
        sigma_lj, dist_pedestrian_mean = self.cal_sigma_lj(x,y,bet_dist,kaisu)

        #合流地点生成プログラム
        #歩行者群の後に引力を発生させる
        #角度差が0°の歩行者群の前•後に引力を発生させて，合流させる．
        goryu = self.cal_goryu(x,y,theta_obst,e,num,dist_pedestrian_mean)

        #歩行者群が楕円の場合，楕円の半径を出力
        r_gr = self.cal_r_gr(x,y,num)

        #速度計算
        vx = -(self.cal_pot(x+self.delt,y,theta_obst,theta_bet,sigma_lj,r_gr,goryu)-self.cal_pot(x,y,theta_obst,theta_bet,sigma_lj,r_gr,goryu))
        vy = -(self.cal_pot(x,y+self.delt,theta_obst,theta_bet,sigma_lj,r_gr,goryu)-self.cal_pot(x,y,theta_obst,theta_bet,sigma_lj,r_gr,goryu))
        v = math.sqrt(vx**2+vy**2)

        vx /= v/self.speed
        vy /= v/self.speed

        print(vx,vy)
        print(goryu)

        return vx,vy


class IdealRobot:

    vx_list = []
    vy_list = []
    num = 0

    def __init__(self,pose,agent=None,color="black"):
        self.pose=pose
        self.r = 0.3
        self.color=color
        self.agent = agent
        self.poses = [pose]
        self.num = True

    def draw(self,ax,elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x,xn],[y,yn],color=self.color)
        c = patches.Circle(xy=(x,y),radius=self.r,fill=False,color=self.color)
        elems.append(ax.add_patch(c))

        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses],[e[1] for e in self.poses],linewidth=0.5,color="black")

    @classmethod
    def state_transition(cls,vx,vy,time,pose):
        return pose + np.array([vx,vy,0])*time

    def one_step(self,time_interval):
        # 速度計算
        pot = LJ()
        #sigma_lj = pot.sigma_lj(self.pose[0],self.pose[1])
        vx,vy = pot.cal_route(self.pose[0],self.pose[1])
        self.pose = self.state_transition(vx,vy,time_interval,self.pose)

        



#歩行者群であるため，楕円で表現
class Obstacle:
    def __init__(self,pose,ellipse,agent=None,color="black"):
        self.pose=pose
        self.r = 0.3
        self.color=color
        self.agent = agent
        self.poses = [pose]
        self.pose_af = pose

        self.width = ellipse[0]*2
        self.height = ellipse[1]*2

    def draw(self,ax,elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        #elems += ax.plot([x,xn],[y,yn],color=self.color)
        #c = patches.Circle(xy=(x,y),radius=self.r,fill=False,color=self.color)
        c = patches.Ellipse(xy=(x,y),width=max(self.width,self.height),height=min(self.width,self.height),angle=theta,fill=False,color=self.color)
        #c = patches.Rectangle(xy=(x-self.width/2,y-self.height/2),width=self.width,height=self.height,angle=theta,fill=False,color=self.color)
        elems.append(ax.add_patch(c))

        self.poses.append(self.pose)
        #elems += ax.plot([e[0] for e in self.poses],[e[1] for e in self.poses],linewidth=0.5,color="black")

    @classmethod
    def state_transition(cls,vx,vy,time,pose):
        return pose + np.array([vx,vy,0])*time

    def step(self,time_interval):
        if not self.agent:
            return
        vx,vy = self.agent.decision()
        self.pose_af = self.state_transition(vx,vy,time_interval,self.pose)


    def one_step(self,time_interval):
        if not self.agent:
            return
        vx, vy = self.agent.decision()
        self.pose = self.state_transition(vx,vy,time_interval,self.pose)

class Between:
    def __init__(self,pose,agent=None,color="black"):
        self.pose=pose
        self.r = 0.3
        self.color=color
        self.agent = agent
        self.poses = [pose]
        self.pose_af = pose


    def draw(self,ax,elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        #elems += ax.plot([x,xn],[y,yn],color=self.color)
        c = patches.Circle(xy=(x,y),radius=self.r,fill=False,color=self.color)
        elems.append(ax.add_patch(c))

        self.poses.append(self.pose)
        #elems += ax.plot([e[0] for e in self.poses],[e[1] for e in self.poses],linewidth=0.5,color="black")

    @classmethod
    def state_transition(cls,vx,vy,time,pose):
        return pose + np.array([vx,vy,0])*time

    def step(self,time_interval):
        if not self.agent:
            return
        vx,vy = self.agent.decision()
        self.pose_af = self.state_transition(vx,vy,time_interval,self.pose)


    def one_step(self,time_interval):
        if not self.agent:
            return
        vx, vy = self.agent.decision()
        self.pose = self.state_transition(vx,vy,time_interval,self.pose)

class Agent:
    def __init__(self,vx,vy):
        self.vx = vx
        self.vy = vy

    def decision(self,observation=None):
        return self.vx,self.vy

class Landmark:
    def __init__(self,x,y):
        self.pos = np.array([x,y]).T
        self.id = None

    def draw(self,ax,elems):
        c = ax.scatter(self.pos[0],self.pos[1],s=100,marker="*",label="landmarks",color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0],self.pos[1],"id:"+str(self.id),fontsize=10))

class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self,landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self,ax,elems):
        for im in self.landmarks:
            im.draw(ax,elems)

world = World(1000,1/10)
straight = Agent(1.0,0)
back = Agent(0,-1)
stop = Agent(0,0)


robot = IdealRobot(np.array([2.3,0.3,0]).T,color="blue")
obs1 = Obstacle(np.array([0.75,2.7,0]).T,[0.8,0.75],straight,color="red")
obs2 = Obstacle(np.array([3.75,2.7,0]).T,[0.8,0.75],straight,color="red")
bet1 = Between(np.array([0.3,2.2,0]).T,straight,color="green")
bet2 = Between(np.array([0.3,3.2,0]).T,straight,color="green")
bet3 = Between(np.array([1.2,2.6,0]).T,straight,color="green")
bet4 = Between(np.array([3.3,2.2,0]).T,straight,color="green")
bet5 = Between(np.array([3.3,3.2,0]).T,straight,color="green")
bet6 = Between(np.array([4.2,2.6,0]).T,straight,color="green")
#bet7 = Between(np.array([13,4.2,0]).T,stop,color="green")

pot = LJ()
pot.obst_append(obs1)
pot.obst_append(obs2)
pot.bet_append(bet1)
pot.bet_append(bet2)
pot.bet_append(bet3)
pot.bet_append(bet4)
pot.bet_append(bet5)
pot.bet_append(bet6)
#pot.bet_append(bet7)

world.robot_append(robot)
world.obj_append(obs1)
world.obj_append(obs2)
world.bet_append(bet1)
world.bet_append(bet2)
world.bet_append(bet3)
world.bet_append(bet4)
world.bet_append(bet5)
world.bet_append(bet6)
#world.bet_append(bet7)

world.draw()


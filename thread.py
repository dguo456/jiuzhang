import time
import random
import threading
import pygame


class Block(threading.Thread):

    def __init__(self, screen):

        #调用父类的构造方法
        super(Block, self).__init__()
        
        #设置方块的贴图
        self.blue = None

        #当前方块坐标
        self.pos = [5*30, -30]

        #设置当前游戏窗口
        self.screen = screen
  
        #当前方块下落的速度
        self.speed = 0.5
        
        #随机获取当前的方块种类
        self.block = BLOCKLIST[random.randint(0, len(BLOCKLIST)-1)]
        
        #当前方块的形状(默认为形状1)
        self.blocktype = 1

        #是否需要产生新的方块(是否碰撞)
        self.__IsNewBlock = False


    def GetIsNewBlock(self):
        return self.__IsNewBlock

    def GetBlock(self):

        #T形状
        if self.block == "TBLOCK":
            if self.blocktype == 1:
                return [(self.pos[0]-30, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0]+30, self.pos[1]), (self.pos[0], self.pos[1]-30)]
            elif self.blocktype == 2:
                return [(self.pos[0], self.pos[1] - 30), (self.pos[0], self.pos[1]), (self.pos[0] + 30, self.pos[1]), (self.pos[0], self.pos[1] + 30)]
            elif self.blocktype == 3:
                return [(self.pos[0] - 30, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0] + 30, self.pos[1]),(self.pos[0], self.pos[1] + 30) ]
            elif self.blocktype == 4:
                return [(self.pos[0] - 30, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] + 30),(self.pos[0], self.pos[1] - 30)]

        #I形状
        elif self.block == "IBLOCK":
            if self.blocktype == 1 or self.blocktype == 3:
                return [(self.pos[0]- 30, self.pos[1]), (self.pos[0], self.pos[1]), (self.pos[0] +30, self.pos[1]), (self.pos[0] + 60, self.pos[1]) ]
            elif self.blocktype == 2 or self.blocktype == 4:
                return [(self.pos[0], self.pos[1] -30), (self.pos[0], self.pos[1]), (self.pos[0], self.pos[1] +30), (self.pos[0], self.pos[1] +60)]

        #O形状
        elif self.block == "OBLOCK":
            return [(self.pos[0] -30, self.pos[1]), self.pos, (self.pos[0] - 30, self.pos[1]-30), (self.pos[0], self.pos[1] - 30)]

        #S形状
        elif self.block == "SBLOCK":
            if self.blocktype == 1 or self.blocktype == 3:
                return [(self.pos[0]+30, self.pos[1]), self.pos, (self.pos[0], self.pos[1] + 30), (self.pos[0]-30, self.pos[1] + 30)]
            elif self.blocktype == 2 or self.blocktype == 4:
                return [(self.pos[0], self.pos[1] - 30), self.pos, (self.pos[0] + 30, self.pos[1]), (self.pos[0] + 30, self.pos[1] + 30)]


        #Z形状
        elif self.block == "ZBLOCK":
            if self.blocktype == 1 or self.blocktype == 3:
                return [(self.pos[0] - 30, self.pos[1]), self.pos, (self.pos[0], self.pos[1]+30), (self.pos[0] + 30, self.pos[1] +30)]
            elif self.blocktype == 2 or self.blocktype == 4:
                return [(self.pos[0], self.pos[1] - 30), self.pos, (self.pos[0] - 30, self.pos[1]), (self.pos[0] - 30, self.pos[1] + 30)]

        #L形状
        elif self.block == "LBLOCK":
            if self.blocktype == 1:
                return [(self.pos[0], self.pos[1]-30), self.pos, (self.pos[0], self.pos[1] + 30), (self.pos[0] + 30, self.pos[1] + 30)]
            elif self.blocktype == 2:
                return [(self.pos[0] - 30, self.pos[1] +30), (self.pos[0] - 30, self.pos[1]), self.pos, (self.pos[0] + 30, self.pos[1])]
            elif self.blocktype == 3:
                return [(self.pos[0] - 30, self.pos[1] - 30), (self.pos[0], self.pos[1] - 30), self.pos, (self.pos[0], self.pos[1] + 30)]
            elif self.blocktype == 4:
                return [(self.pos[0] -30, self.pos[1]), self.pos, (self.pos[0] + 30, self.pos[1]), (self.pos[0]+30, self.pos[1]-30)]

        #J形状
        elif self.block == "JBLOCK":
            if self.blocktype == 1:
                return [(self.pos[0], self.pos[1]-30), self.pos, (self.pos[0], self.pos[1] + 30), (self.pos[0]-30, self.pos[1] + 30)]
            elif self.blocktype == 2:
                return [(self.pos[0] - 30, self.pos[1] - 30), (self.pos[0] - 30, self.pos[1]), self.pos, (self.pos[0] + 30, self.pos[1])]
            elif self.blocktype == 3:
                return [(self.pos[0] + 30, self.pos[1] - 30), (self.pos[0], self.pos[1] - 30), self.pos, (self.pos[0], self.pos[1] + 30)]
            elif self.blocktype == 4:
                return [(self.pos[0] - 30, self.pos[1]), self.pos, (self.pos[0] + 30, self.pos[1]), (self.pos[0] + 30, self.pos[1] + 30)]

        return None

    def Move(self):
        for event in pygame.event.get():
            # 监听退出事件
            if event.type == pygame.QUIT:
                OVER = True
                exit()
            # 监听键盘事件
            elif event.type == pygame.KEYDOWN:
                # 形状改变事件
                if event.key == pygame.K_UP:
                    # 预测下当形状发生改变时， 会不会与存在的方块重复， 如果会则禁止改变形状
                    currentType = self.blocktype
                    IsMove = True

                    # 先改变当前的形状， 1 --> 2 --> 3 --> 4, 4 --> 1
                    self.blocktype = 1 if (self.blocktype == 4) else self.blocktype + 1

                    for w, h in self.GetBlock():
                        if (GlobalDict[w / 30, h / 30] == 1):
                            IsMove = False
                            break

                    # 如果禁止改变形状， 则变回原来的形状
                    self.blocktype = self.blocktype if (IsMove) else currentType

                # 方块快速下落事件
                if event.key == pygame.K_DOWN:
                    # 改变子线程的阻塞时间， 即可实现方块的快速下落
                    self.speed = 0.01

                # 左右移动事件
                if event.key == pygame.K_LEFT:
                    # 预测方块向左移动时是否存在方块， 如果为True则无法移动
                    IsMove = True
                    for w, h in self.GetBlock():
                        if (GlobalDict[w / 30 - 1, h / 30] == 1):
                            IsMove = False
                            break

                    if (IsMove):
                        self.pos[0] -= 30

                elif event.key == pygame.K_RIGHT:
                    # 预测方块向右移动时是否存在方块， 如果为True则无法移动
                    IsMove = True
                    for w, h in self.GetBlock():
                        if (GlobalDict[w / 30 + 1, h / 30] == 1):
                            IsMove = False
                            break

                    if (IsMove):
                        self.pos[0] += 30


    def Collition(self):
        global IsNewBlock, GAMEOVER

        #遍历当前方块的x坐标和y坐标
        for w, h in self.GetBlock():
            #预测当前方块对象是否触碰到其他方块
            if(GlobalDict[w/30, h/30 +1]  == 1):
                #判断是否有方块超出顶部
                if(h/30 < 0):
                    #游戏结束
                    GAMEOVER = True
                    break

                #遍历每块方块， 在数据字典中记录其坐标为1
                for w, h in self.GetBlock():
                    GlobalDict[int(w / 30), int(h / 30)] = 1

                #记录发生碰撞
                self.__IsNewBlock = True
                break

    def Update(self):
        for b in self.GetBlock():
            self.screen.blit(self.blue, b)

    def run(self):
        global GAMEOVER

        while True:
            # 当游戏结束时， 结束该子线程
            if GAMEOVER:
                break

            # 碰撞检测
            self.Collition()
            # 如果发生了碰撞,则结束该子线程
            if self.GetIsNewBlock():
                break

            self.pos[1] += 30
            # 控制方块下落速度
            time.sleep(self.speed)

def FlushScreen(screen, blue):
    #满的方块判断
    for h in range(20):
        #从最底层开始， 一行行的判断
        c_h = 19 - h #当前行数
        count = 0
        for w in range(10):
            count += GlobalDict[w, c_h]
        #当这一行的方块满了, 重置这一行的值
        if count == 10:
            for w in range(10):
                GlobalDict[w, c_h] = 0

            #将这一行上面的方块全部下降一格
            for h1 in range(c_h):
                c_h1 = c_h - 1 - h1 #当前行数
                for w1 in range(10):
                    if GlobalDict[w1, c_h1] == 1:
                        #下降一格
                        GlobalDict[w1, c_h1 + 1] = 1
                        #当前坐标的方块值清空
                        GlobalDict[w1,  c_h1] = 0

    #将数据字典中值为1的坐标填充方块
    for h in range(20):
        for w in range(10):
            if GlobalDict[w, h] == 1:
                screen.blit(blue, (w*30,h*30))



'''全局变量'''
#俄罗斯方块的数据字典
GlobalDict = {}
#判断游戏是否结束
GAMEOVER = True
#声明一个方块形状列表
BLOCKLIST = ["TBLOCK", "SBLOCK", "ZBLOCK", "IBLOCK", "OBLOCK", "JBLOCK", "LBLOCK"]
# 初始化pygame
pygame.init()
# 设置一个字体模块
font = pygame.font.SysFont("MicrosoftYaHei", 36)
# 计算该字体所占一行的高度
font_height = font.get_linesize()

def main():
    global GlobalDict, GAMEOVER
    #创建一个screen
    screen = pygame.display.set_mode((600, 600), 0, 32)
    # 设置贴图对象
    blue = pygame.image.load("img/blue.png").convert()
    #设置游戏帧率
    # GameFrame(60)
    # 新建一个方块对象
    BLOCK = Block(screen)

    #游戏主线程
    while True:
        #游戏进行时
        if not GAMEOVER:
            #每当需要新建一个新的方块时需要执行的语句块
            if(BLOCK.GetIsNewBlock()):
                # 创建一个新的方块对象
                BLOCK = Block(screen)
                # 启动方块对象的子线程
                BLOCK.start()

            #绘制游戏界面
            GameScreen(screen, )
            # 方块的移动检测
            BLOCK.Move()
            #更新当前方块的位置
            BLOCK.Update()
            #刷新游戏界面
            FlushScreen(screen, blue)

        #当游戏结束时
        else:
            screen.fill((255, 255, 255))
            screen.blit(font.render("Please Enter 'SPACE' Start The Game", True, (0, 0, 0)), (50, 250))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.KEYDOWN:
                    print(event.key)
                    if event.key == pygame.K_SPACE:
                        GAMEOVER = False
                        initGame()
                        # 新建一个方块对象
                        BLOCK = Block(screen)
                        # 开始当前方块对象的子线程
                        BLOCK.start()

        #将贴图更新到主程序中
        pygame.display.update()


def GameScreen(screen, ):
    '''绘制游戏界面的方法'''
    global font, font_height
    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 0, 255), (301, 0), (301, 600))

def initGame():
    '''初始化俄罗斯方块数据字典'''
    
    #游戏视图内坐标
    for h in range(20):
        for w in range(10):
            GlobalDict[w, h] = 0
    
    #边界
    for h in range(23):
        GlobalDict[-1, h-2] = 1
        GlobalDict[10, h-2] = 1
        
    #边界， 这里顶部设置为无边界， 防止刚生成方块就游戏结束
    for w in range(14):
        GlobalDict[w-2, -1] = 0
        GlobalDict[w-2, 20] = 1





if __name__ == '__main__':
    main()
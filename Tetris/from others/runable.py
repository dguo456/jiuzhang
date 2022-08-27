from processing import *
import random
import time
#code is mess please organize :(

#Blocks make up peices.
class Block:
    def __init__(self):
        self.r = 0
        self.g = 0
        self.b = 0  
        self.value = 0

#values!
gridheight = 23 #how tall grid is
gridwidth = 11 #how wide grid is
grid = [[Block() for j in range(gridwidth)] for i in range(gridheight)] #making grid
blocksize = 20
startx = 100 
starty = 450 
spawnx = 4 #where peice spawns x wise
spawny = 22 #where piece spawns y wise
changex = 0 #how far away piece has moved from spawnx
changey = 0 #how far away piece has moved from spawny
starttime = time.time() 
gamestarttime = time.time()
piecex = 0 #peice's curent x location
piecey = 0 #peice's curent y location
dropspeed = 1 #how fast you fall
score = 0 #total score
delay = 1.1 #time between falling
lastchangey = 0

#class of all pieces
class Piece:
      
    def __init__(self):
        self.blocks = None
        self.all_blocks = None
        self.curr_index = 0
        self.r = 0
        self.g = 0
        self.b = 0
  
    def nextblock(self):
        self.curr_index = (self.curr_index + 1) % len(self.all_blocks)
        self.blocks = self.all_blocks[self.curr_index]
    

# all peices are pushed to bottom left corner in the grid
# L peice and possible rotations
class Lpiece(Piece):
      
  def __init__(self):
    super().__init__(self)
    #colors
    self.r = 66
    self.g = 134
    self.b = 244
    
    block1 = [ [0 for j in range(3)] for i in range(3)]
    block1[0][0] = 1
    block1[1][0] = 1
    block1[0][1] = 1
    block1[0][2] = 1
    
    
    block2 = [ [0 for j in range(3)] for i in range(3)]
    block2[0][1] = 1
    block2[0][2] = 1
    block2[1][2] = 1
    block2[2][2] = 1
    
    
    block3 = [ [0 for j in range(3)] for i in range(3)]
    block3[1][2] = 1
    block3[2][2] = 1
    block3[2][1] = 1
    block3[2][0] = 1
    
    
    block4 = [ [0 for j in range(3)] for i in range(3)]
    block4[0][0] = 1
    block4[1][0] = 1
    block4[2][0] = 1
    block4[2][1] = 1    
    
    
    self.all_blocks = [block1, block2, block3, block4]
    self.blocks = self.all_blocks[0]
  
  

    
#O peice and possible rotations   
class Opiece(Piece):
  def __init__(self):
    super().__init__(self)
    #colors
    self.r = 143
    self.g = 244
    self.b = 65
    
    block1 = [ [0 for j in range(4)] for i in range(4)]
    block1[1][1] = 1
    block1[1][2] = 1
    block1[2][1] = 1
    block1[2][2] = 1
    
    
    self.all_blocks = [block1]
    self.blocks = self.all_blocks[0]
  
  
#I peice and possible rotations  
class Ipiece(Piece):
  def __init__(self):
    super().__init__(self)
    #colors
    self.r = 241
    self.g = 252
    self.b = 80
  
    block1 = [ [0 for j in range(4)] for i in range(4)]
    block1[1][0] = 1
    block1[1][1] = 1
    block1[1][2] = 1
    block1[1][3] = 1
    
    
    block2 = [ [0 for j in range(4)] for i in range(4)]
    block2[0][2] = 1
    block2[1][2] = 1
    block2[2][2] = 1
    block2[3][2] = 1
    
    
    block3 = [ [0 for j in range(4)] for i in range(4)]
    block3[2][0] = 1
    block3[2][1] = 1
    block3[2][2] = 1
    block3[2][3] = 1
  
  
    block4 = [ [0 for j in range(4)] for i in range(4)]
    block4[0][1] = 1
    block4[1][1] = 1
    block4[2][1] = 1
    block4[3][1] = 1
    
    
    self.all_blocks = [block1, block2, block3, block4]
    self.blocks = self.all_blocks[0]


#T peice and possible rotations
class Tpiece(Piece):
  def __init__(self):
    super().__init__(self)
    #colors
    self.r = 242
    self.g = 206
    self.b = 62
    
    block1 = [ [0 for j in range(3)] for i in range(3)]
    block1[0][0] = 1
    block1[1][0] = 1
    block1[1][1] = 1
    block1[2][0] = 1
    
    block2 = [ [0 for j in range(3)] for i in range(3)]
    block2[0][0] = 1
    block2[0][1] = 1
    block2[0][2] = 1
    block2[1][1] = 1 
    
    
    block3 = [ [0 for j in range(3)] for i in range(3)]
    block3[2][2] = 1
    block3[1][2] = 1
    block3[0][2] = 1
    block3[1][1] = 1 
    
    block4 = [ [0 for j in range(3)] for i in range(3)]
    block4[2][1] = 1
    block4[2][0] = 1
    block4[2][2] = 1
    block4[1][1] = 1 
    
    self.all_blocks = [block1, block2, block3, block4]
    self.blocks = self.all_blocks[0]

#S peice and possible rotations
class Spiece(Piece):
  def __init__(self):
    super().__init__(self)
    #colors
    self.r = 239
    self.g = 71
    self.b = 59
    
    block1 = [ [0 for j in range(3)] for i in range(3)] 
    block1[0][0] = 1
    block1[1][0] = 1
    block1[1][1] = 1
    block1[2][1] = 1
    
    block2 = [ [0 for j in range(3)] for i in range(3)] 
    block2[1][0] = 1
    block2[1][1] = 1
    block2[0][2] = 1
    block2[0][1] = 1
    
    block3 = [ [0 for j in range(3)] for i in range(3)] 
    block3[0][1] = 1
    block3[1][2] = 1
    block3[1][1] = 1
    block3[2][2] = 1
    
    block4 = [ [0 for j in range(3)] for i in range(3)] 
    block4[2][0] = 1
    block4[2][1] = 1
    block4[1][1] = 1
    block4[1][2] = 1
    
    
    self.all_blocks = [block1, block2, block3, block4]
    self.blocks = self.all_blocks[0]
    
#Z peice and possible rotations
class Zpiece(Piece):
  def __init__(self):
    super().__init__(self)
    #colors
    self.r = 178
    self.g = 57
    self.b = 239
    
    block1 = [ [0 for j in range(3)] for i in range(3)]
    block1[0][1] = 1
    block1[1][0] = 1
    block1[1][1] = 1
    block1[2][0] = 1
    
    block2 = [ [0 for j in range(3)] for i in range(3)]
    block2[0][0] = 1
    block2[0][1] = 1
    block2[1][1] = 1
    block2[1][2] = 1
    
    block3 = [ [0 for j in range(3)] for i in range(3)]
    block3[0][2] = 1
    block3[1][1] = 1
    block3[1][2] = 1
    block3[2][1] = 1
  
    block4 = [ [0 for j in range(3)] for i in range(3)]
    block4[1][0] = 1
    block4[1][1] = 1
    block4[2][1] = 1
    block4[2][2] = 1
    
    self.all_blocks = [block1, block2, block3, block4]
    self.blocks = self.all_blocks[0]

#J peice and possible rotations
class Jpiece(Piece):
  def __init__(self):
    super().__init__(self)
    #colors
    self.r = 249
    self.g = 94
    self.b = 207
    
    block1 = [ [0 for j in range(3)] for i in range(3)]
    block1[0][0] = 1
    block1[0][1] = 1
    block1[1][0] = 1
    block1[2][0] = 1

    block2 = [ [0 for j in range(3)] for i in range(3)]
    block2[0][0] = 1
    block2[0][1] = 1
    block2[0][2] = 1
    block2[1][2] = 1
    
    block3 = [ [0 for j in range(3)] for i in range(3)]
    block3[0][2] = 1
    block3[1][2] = 1
    block3[2][2] = 1
    block3[2][1] = 1
    
    block4 = [ [0 for j in range(3)] for i in range(3)]
    block4[1][0] = 1
    block4[2][2] = 1
    block4[2][1] = 1
    block4[2][0] = 1
    
    self.all_blocks = [block1, block2, block3, block4]
    self.blocks = self.all_blocks[0]

#pick a random piece from all possible pieces.
def randompiece():
  
  piecearray = ["S","T","L","Z","J","O","I"]
  randomindex = random.randint(0, len(piecearray)-1)
  piece = piecearray[randomindex]
  if piece == "S":
    return Spiece()
  elif piece == "T":
    return Tpiece()
  elif piece == "L":
    return Lpiece()
  elif piece == "Z":
    return Zpiece()
  elif piece == "J":
    return Jpiece()
  elif piece == "I":
    return Ipiece()
  elif piece == "O":
    return Opiece()
  
#figure out how much time passes between falling
def calcdelay(delay,t):
  return 1 / (delay ** (0.1 * t))
  
  
#size of screen
def setup():
    size(500,500)
    noStroke()
  
def draw():
  global changey, starttime, piecey, currentpiece, changex, changey

  currenttime = time.time()
  
  clear()
  
  drawgrid()
  fill(0, 0, 0)
  textAlign(RIGHT, BASELINE)
  text(score,480,20)
  drawpiece()
  timesincestart = currenttime - gamestarttime
  if currenttime - starttime >= dropspeed * calcdelay(delay,timesincestart):
    if checkdown(currentpiece,changex,changey):
      currentpiece = None
      checkline()
      currentpiece = randompiece()
      changex = 0
      changey = 0
      if lastchangey == -1:
        textSize(42)
        textAlign(CENTER,CENTER)
        fill(10, 11, 13)
        text("You Lose!",220,220)
        text("Final Score:",220,255)
        textAlign(LEFT,CENTER)
        text(score,330,255)
        exit()
    starttime = currenttime
    deletepiece(currentpiece)
    changey = changey - 1
    setpiece(currentpiece)

  

#checks for object\floor underneath, and stops falling if there is floor
def checkdown(piece, changex, changey):
    global grid, piecex, piecey, lastchangey
    #-----
    columns = range(len(piece.blocks))
    for col in range(len(piece.blocks)):
        for i in columns:
            if piece.blocks[i][col] == 1:
                if changey + spawny + col <= 0 or grid[changey + spawny + col - 1][changex + spawnx + i].value == 1:
                    lastchangey = changey
                    return True
                columns.remove(i)
    return False

#makes piece hit left wall
def checkleft(piece,changex,changey):  
  global grid, piecex, piecey, col, gridwidth
  rows =[0,1,2]
  for i in range(len(piece.blocks)):
    for row in rows:
      if piece.blocks[i][row] == 1:
        if changex + spawnx + i <= 0 or changey + spawny + row < gridheight and grid[changey + spawny + row][changex + spawnx + i - 1].value:
          return False
        rows.remove(row)

  return True

#makes piece hit right wall
def checkright(piece, changex, changey):  
  global grid, piecex, piecey, col, gridwidth
  rows = [0,1,2]
  for i in range(len(piece.blocks) - 1, -1, -1):
    for row in rows:
      if piece.blocks[i][row] == 1:
        if changex + spawnx + i + 1 >= gridwidth or changey + spawny + row < gridheight and grid[changey + spawny + row][changex + spawnx + i + 1].value:
          return False
        rows.remove(row)
        continue
      
  return True

#make sure piece doesn't go through left wall
def checkoutleft(piece,changex,changey):  
  global grid, piecex, piecey, col, gridwidth
  rows =[0,1,2]
  for i in range(len(piece.blocks)):
    for row in rows:
      if piece.blocks[i][row]  == 1:
        if changex + spawnx + i < 0:
          return False
        rows.remove(row)
        continue
      
  return True 
  
#make sure peice doesn't go through right wall
def checkoutright(piece, changex, changey):  
  global grid, piecex, piecey, col, gridwidth
  rows = [0,1,2]
  for i in range(len(piece.blocks) - 1, -1 , -1):
    for row in rows:
      if piece.blocks[i][row]  == 1:
        if changex + spawnx + i >= gridwidth:
          return False
        rows.remove(row)
        continue
      
  return True

#make sure peices don't fall through floor
def checkoutdown(piece,changex,changey):  
  global grid, piecex, piecey, col, gridwidth
  rows = [0,1,2]
  for i in range(len(piece.blocks)):
    for row in rows:
      if piece.blocks[i][row]  == 1:
        if changey + spawny + i < 0:
          return False
        rows.remove(row)
        continue
  return True



#check if you can clear a line
def checkline():
  global grid
  lines = []
  offsets = []
  currentoffset = 0
  lastfilledline = 0
  for row in range(len(grid)):
    line = True
    for col in range(len(grid[row])):
      if grid[row][col].value == 0:
        line = False
        if row - lastfilledline >= 1:
          offsets.append(row - lastfilledline)
          lastfilledline = row
        break
        
    if line:
      for col in range(len(grid[row])):
        grid[row][col].value = 0
      if len(lines) == 0 or row == lastfilledline:
        lines.append(row)
    else:
      lastfilledline = lastfilledline + 1
      if lines and lines[len(lines)-1] - lastfilledline >= 4:
        offsets.append(4)
  if len(lines):
    checkscore(offsets)
    for i in range(len(lines)):
      offset = 1
      if len(offsets) >= 1:
        offset = offsets[i]
      for row in range(lines[i] - i,gridheight-offset):
        for col in range(len(grid[row])):
            grid[row][col].value = grid[row+offset][col].value
            grid[row][col].r = grid[row+offset][col].r
            grid[row][col].g = grid[row+offset][col].g
            grid[row][col].b = grid[row+offset][col].b
            
            
            #make peice stop exitsting, and make it white
            grid[row+offset][col].value = 0
            grid[row+offset][col].r = 255
            grid[row+offset][col].g = 255
            grid[row+offset][col].b = 255

#set peice to sit still and keep it's color
def setpiece(piece):
  global spawnx, spawny, grid
  piecex = spawnx
  piecey = spawny
  
  for i in range(len(piece.blocks)):
    for j in range(len(piece.blocks[i])):
      if (spawnx + i + changex < gridwidth and spawnx + i + changex >= 0) and (spawny + j + changey < gridheight and spawny + j + changey >= 0) and piece.blocks[i][j]:
        
        grid[spawny + j + changey][spawnx + i + changex].value = 1
        grid[spawny + j + changey][spawnx + i + changex].r = piece.r
        grid[spawny + j + changey][spawnx + i + changex].g = piece.g
        grid[spawny + j + changey][spawnx + i + changex].b = piece.b


#make piece dissapear
def deletepiece(piece):
  global spawnx, spawny, grid
  piecex = spawnx
  piecey = spawny
  
  for i in range(len(piece.blocks)):
    for j in range(len(piece.blocks[i])):
      if (spawnx + i + changex  < gridwidth and spawnx + i >= 0) and (spawny + j + changey < gridheight and spawny + j >= 0) and piece.blocks[i][j]:
        
        grid[spawny + j + changey][spawnx + i + changex].value = 0

def clear():
  fill(255,255,255)
  noStroke()
  rect(0,0,500,500,3)
  
def drawgrid():
  global startx, starty, blocksize, grid
  stroke(0,0,0)

  for y in range (len(grid) - 1 , -1, -1):
    for x in range(len(grid[y])):
      if grid[y][x].value:
        fill(grid[y][x].r,grid[y][x].g,grid[y][x].b)
      else:
        fill(255)
      rect(startx + x * blocksize,starty - y * blocksize ,blocksize,blocksize)


def drawpiece():
  fill(color(currentpiece.r,currentpiece.g,currentpiece.b))
  for j in range(len(currentpiece.blocks) - 1, -1, -1): 
    for i in range(len(currentpiece.blocks[j])): 
      if currentpiece.blocks[j][i] == 1:
        
        rect(startx + (j+changex + spawnx ) * blocksize ,starty - (i+changey + spawny) * blocksize,blocksize,blocksize)

#calculate score for ammount of lines cleared
def checkscore(lines):
  global score
  n = sum(lines)
  if n == 1:
    score = score + 40
  if n == 2:
    score = score + 100
  if n == 3:
    score = score + 300
  if n == 4:
    score = score + 1200



  
def keyPressed():
    global changex, piecex, dropspeed, grid, columns, changey

#move left
    if keyCode == LEFT:
      if checkleft(currentpiece,changex,changey):
        deletepiece(currentpiece)
        changex = changex - 1
        piecex = piecex -1
        setpiece(currentpiece)
    
#move right
    if keyCode == RIGHT:
      if checkright(currentpiece,changex,changey):
        deletepiece(currentpiece)
        changex = changex + 1
        piecex = piecex +1
        setpiece(currentpiece)
      
#fall fast    
    if keyCode == DOWN:
      dropspeed = 0.1

#rotate piece
    if keyCode == UP:
      deletepiece(currentpiece)
      currentpiece.nextblock()
      while not checkoutright(currentpiece,changex,changey):
        changex = changex - 1
        
      while not checkoutleft(currentpiece,changex,changey):
        changex = changex + 1
        
      while not checkoutdown(currentpiece,changex,changey):
        changey = changey + 1
        
      setpiece(currentpiece)
      
#immediantly fall (broken)
'''   if key == " ":
      while not checkdown(currentpiece, changex, changey):
        deletepiece(currentpiece)
        changey = changey - 1
        setpiece(currentpiece)'''

#stop falling fast
def keyReleased():
  global dropspeed
  
  if keyCode == DOWN:
    dropspeed = 1


currentpiece = randompiece()
setpiece(currentpiece)

run()
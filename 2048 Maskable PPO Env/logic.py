# logic to be imported in the main file
import random
import numpy as np
import math

# function to initialize game / grid at the start
def start_game():
    # declaring an empty list then appending 4 list each with four elements as 0
    mat =[]
    for i in range(4):
        mat.append([0] * 4)
 
    # calling the function to add a new 2or 4 in grid after every step
    add_new_2_or_4(mat)
    #game_print(mat)
    return mat
 
# function to add a new 2 or 4 in grid at any random empty cell
def add_new_2_or_4(mat):
    # choosing a random index for row and column
    r = random.randint(0, 3)
    c = random.randint(0, 3)

    while(mat[r][c] != 0):
        r = random.randint(0, 3)
        c = random.randint(0, 3)
 
    # we will place a 2 or 4 (with 10%chance) at that empty random cell
    if random.random() < 0.10:
        mat[r][c] = 4
    else:
        mat[r][c] = 2
 
# function to get the current state of game
def get_current_state(mat):
 
    # if we are still left with atleast one empty cell game is not yet over
    for i in range(4):
        for j in range(4):
            if(mat[i][j]== 0):
                return 'GAME NOT OVER'
 
    ''' or if no cell is empty now
    but if after any move left, right, up or down, if any two cells
    gets merged and create an empty cell then also game is not yet over '''
    for i in range(3):
        for j in range(3):
            if(mat[i][j]== mat[i + 1][j] or mat[i][j]== mat[i][j + 1]):
                return 'GAME NOT OVER'
 
    for j in range(3):
        if(mat[3][j]== mat[3][j + 1]):
            return 'GAME NOT OVER'
    for i in range(3):
        if(mat[i][3]== mat[i + 1][3]):
            return 'GAME NOT OVER'
 
    # else we have lost the game
    return 'LOST'

# all the functions defined below are for left swap initially
 
# function to compress the grid after every step before and after merging cells
def compress(mat):
    # bool variable to determine any change happened or not
    changed = False
    # empty grid
    new_mat = []
    
    cells_moved = 0 #apply penalty for each cell moved
    
    # with all cells empty
    for i in range(4):
        new_mat.append([0] * 4)
         
    # here we will shift entries of each cell to it's extreme left row by row
    # loop to traverse rows
    for i in range(4):
        pos = 0
 
        # loop to traverse each column in respective row
        for j in range(4):
            if(mat[i][j] != 0):
                # if cell is non empty then we will shift it's number to
                # previous empty cell in that row denoted by pos variable
                new_mat[i][pos] = mat[i][j]
                 
                if(j != pos):
                    changed = True
                    cells_moved +=1
                pos += 1
    if cells_moved != 0:
        cells_moved = math.log2(cells_moved) 
    # returning new compressed matrix and the flag variable
    return new_mat, changed, cells_moved
 
# function to merge the cells in matrix after compressing
def merge(mat):
    changed = False
    reward_list = [0]
    
    for i in range(4):
        for j in range(3):
            # if current cell has same value as next cell in the row and they
            # are non empty then
            if(mat[i][j] == mat[i][j + 1] and mat[i][j] != 0):
                # double current cell value and empty the next cell
                mat[i][j] = mat[i][j] * 2
                
                #set reward for gym env !!!
                reward_list.append(mat[i][j])
                
                mat[i][j + 1] = 0
                # make bool variable True indicating the new grid after
                # merging is different
                changed = True

    #apply log2 to reward
    log_reward = [math.log2(item) for item in reward_list if item !=0]
    
    #reward is sum of log2 values of all merges of the selected action, play around with the reward function for better results
    merge_reward = sum(log_reward)
    #merge_reward = max(log_reward)

    return mat, changed, merge_reward
 
# function to reverse the matrix means reversing the content of each row
# (reversing the sequence)
def reverse(mat):
    new_mat =[]
    for i in range(4):
        new_mat.append([])
        for j in range(4):
            new_mat[i].append(mat[i][3 - j])
    return new_mat
 
# function to get the transpose of matrix means interchanging rows and column
def transpose(mat):
    new_mat = []
    for i in range(4):
        new_mat.append([])
        for j in range(4):
            new_mat[i].append(mat[j][i])
    return new_mat
 
# function to update the matrix if we move / swipe left
def move_left(grid):
    # first compress the grid
    new_grid, changed1, cells_moved = compress(grid)   
    # then merge the cells
    new_grid, changed2, left_reward = merge(new_grid)
    changed = changed1 or changed2
    # again compress after merging
    new_grid, temp, merged_cells = compress(new_grid)

    #apply penalty to reward
    #left_reward = left_reward - cells_moved
    
    # return new matrix and bool changed
    return new_grid, changed, left_reward
 
# function to update the matrix if we move / swipe right
def move_right(grid):
    # to move right we just reverse the matrix
    new_grid = reverse(grid)
    # then move left
    new_grid, changed, right_reward = move_left(new_grid)
    # then again reverse matrix will give us desired result
    new_grid = reverse(new_grid)
    return new_grid, changed, right_reward
 
# function to update the matrix if we move / swipe up
def move_up(grid):
    # to move up we just take transpose of matrix
    new_grid = transpose(grid)
    # then move left (calling all included functions) then
    new_grid, changed, up_reward = move_left(new_grid)
    # again take transpose will give desired results
    new_grid = transpose(new_grid)
    return new_grid, changed, up_reward
 
# function to update the matrix if we move / swipe down
def move_down(grid):
    # to move down we take transpose
    new_grid = transpose(grid)
    # move right and then again
    new_grid, changed, down_reward = move_right(new_grid)
    # take transpose will give desired results
    new_grid = transpose(new_grid)
    return new_grid, changed, down_reward

#function to diplay mat in game-like format
def game_print(mat):
    for row in mat: print(*row)

#functions to create the masks for invalid move masking
def check_left(mat):
    problem_rows=0
    mat = np.array(mat)
    def check_row(row):
        problem_row=0
        if row[1]==0 and row[2]==0 and row[3]==0:
            problem_row=1
        elif row[0]!=row[1] and row[1]!=row[2] and row[2]!=row[3] and row[1]!=0 and row[2]!=0 and row[0]!=0:
            problem_row=1
        elif row[0]!=row[1] and row[2]==row[3]==0 and row[0]!=0:
            problem_row=1
        return problem_row
    
    for i in range(4):
        problem_row = check_row(mat[i])
        problem_rows+= problem_row
    if problem_rows == 4:
        return False
    else:
        return True
        
def check_right(mat):
    mat = reverse(mat)
    flag = check_left(mat)
    return flag

def check_up(mat):
    mat = transpose(mat)
    flag = check_left(mat)
    return flag

def check_down(mat):
    mat = transpose(mat)
    flag = check_right(mat)
    return flag
    
    

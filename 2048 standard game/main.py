# 2048.py
# importing the logicv1.py file where we have written all the logic functions used
import logicv1
import gui
import numpy as np
 
# Driver code
if __name__ == '__main__':
    # calling start_game function to initialize the matrix
    mat = logicv1.start_game()
    #gui.gui(mat)
status = 'GAME NOT OVER'

while(True):
    status = logicv1.get_current_state(mat)
    if status != 'GAME NOT OVER':
        print(status)
        break
    
    # taking the user input for next step
    x = input("Press the command : ")

    #gui.gui(mat)
    
    # we have to move up
    if(x == 'W' or x == 'w'):

        if not logicv1.check_up(mat):
            print('action not possible, enter new action')
            continue
        
        # call the move_up function
        mat, flag = logicv1.move_up(mat)
        # get the current state and print it
        status = logicv1.get_current_state(mat)
        print(status)
        # if game not over then continue and add new number
        if(status == 'GAME NOT OVER'):
            logicv1.add_new_2_or_4(mat)
        # else break the loop
        else:
            break
 
    # the above process will be followed in case of each type of move below
 
    # to move down
    elif(x == 'S' or x == 's'):

        if not logicv1.check_down(mat):
            print('action not possible, enter new action')
            continue
        
        mat, flag = logicv1.move_down(mat)
        status = logicv1.get_current_state(mat)
        print(status)
        if(status == 'GAME NOT OVER'):
            logicv1.add_new_2_or_4(mat)
        else:
            break
 
    # to move left
    elif(x == 'A' or x == 'a'):

        if not logicv1.check_left(mat):
            print('action not possible, enter new action')
            continue
        
        mat, flag = logicv1.move_left(mat)
        status = logicv1.get_current_state(mat)
        print(status)
        if(status == 'GAME NOT OVER'):
            logicv1.add_new_2_or_4(mat)
        else:
            break
 
    # to move right
    elif(x == 'D' or x == 'd'):

        if not logicv1.check_right(mat):
            print('action not possible, enter new action')
            continue
        
        mat, flag = logicv1.move_right(mat)
        status = logicv1.get_current_state(mat)
        print(status)
        if(status == 'GAME NOT OVER'):
            logicv1.add_new_2_or_4(mat)
        else:
            break
    else:
        print("Invalid Key Pressed")
 
    # print the matrix after each move
    logicv1.game_print(mat)

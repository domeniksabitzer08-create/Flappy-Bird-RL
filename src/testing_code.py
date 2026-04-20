import pygame

from flappy_bird_env import *



def main():
    action = 0
    is_done = False
    Env = FlappyBirdEnv(1,True)

    while not is_done:
        action = Env.get_user_input()
        #print(f"action: {action}")ssssssss
        Env.step(action)




if __name__ == "__main__":
    main()
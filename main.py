import pygame
import torch
import rainforce_train_main
import pandas as pd
import numpy as np
from tkinter import *


#win  = Tk()
#win.geometry('600x800')
#win.title("입력")


csv = pd.read_csv("score.csv", sep = ",",encoding='utf-8-sig')
df = pd.DataFrame(csv)



# 게임 화면 크기 및 색상 설정
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
clock = pygame.time.Clock()
# 화면 업데이트 함수 정의
def render(env, screen, score):
    screen.fill(BLACK)
    pygame.draw.circle(screen, WHITE, (int(env.ball_x), int(env.ball_y)), env.ball_radius)
    pygame.draw.rect(screen, WHITE, (env.paddle_x, 0, env.paddle_width, env.paddle_height)) 
    pygame.draw.rect(screen, WHITE, (env.player_paddle_x, 620, env.paddle_width, env.paddle_height)) 
    
    font = pygame.font.Font(None, 36)
    score_text = font.render(f'Score: {score}', True, WHITE)
    #screen.blit(score_text, (10, 10)) 

    pygame.display.flip()

# DQN 에이전트 설정 (모델 로드)
class DQNAgent:
    def __init__(self, input_dim, output_dim, model_path):
        self.q_network = rainforce_train_main.DQN(input_dim, output_dim)
        self.q_network.load_state_dict(torch.load(model_path))
        self.q_network.eval()  

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():  
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

# 게임 환경 초기화
env = rainforce_train_main.PongEnv()
agent = DQNAgent(input_dim=3, output_dim=3, model_path='dqn_model_final1.pth')  
num_episodes = 10  

# Pygame 초기화

player_paddle_speed = 10  
env.player_paddle_x = env.width / 2 - env.paddle_width / 2 

while True:
    score = 0
    game = input("게임을 시작하시겠습니까?(y/n) : ")
    if game == "y":
        name = input("닉네임을 입력해주세요 : ")
        pygame.init()
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("DQN Ping-Pong Game")
        clock = pygame.time.Clock() 
        fps = 120  
        player_score = 0  
        ai_score = 0
        result = 0
        
        for i in range(3):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT] and env.player_paddle_x > 0:
                    env.player_paddle_x -= player_paddle_speed
                if keys[pygame.K_RIGHT] and env.player_paddle_x < env.width - env.paddle_width:
                    env.player_paddle_x += player_paddle_speed
                
                action = agent.choose_action(state)
                next_state, reward, done, ai_reward, player_reward = env.step(action)
                total_reward += reward
                
                if ai_reward == -1:
                    player_score +=1
                if player_reward == -1:
                    ai_score += 1

                
            
                state = next_state

                render(env, screen, score)  
                
                clock.tick(fps)


        
        
        if player_score > ai_score:
                result = 1
        elif player_score <= ai_score:
                result = 0
        print(player_score)
        print(ai_score)
        
        print(result)
        time = int(int(pygame.time.get_ticks()) / 1000)
        print(time)
        data = pd.DataFrame({'이름': [name], '승패': [result], '시간': [time]})
        df = pd.concat([df, data], ignore_index=True)
        df = df.sort_values(by='시간', ascending=False)
        
        df.to_csv('score.csv', index=False)
        print("데이터가 성공적으로 저장되었습니다.")


    else:
        break
    #print("Tick : {}초".format( int(int(pygame.time.get_ticks()) / 1000)))

    #print(f"Episode {episode + 1}, Total Reward: {total_reward}, Score: {score}")


pygame.quit()

import pygame
import rainforce_train_2
import torch
#게임 화면 크기 및 색상 설정
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
clock = pygame.time.Clock()

#화면 업데이트 함수 정의
def render(env, screen, episode):
    screen.fill(BLACK)

    pygame.draw.circle(screen, WHITE, (int(env.ball_x), int(env.ball_y)), env.ball_radius)
    pygame.draw.rect(screen, WHITE, (env.paddle_x, env.paddle_height, env.paddle_width, env.paddle_height))
    episode_text = myFont.render(f"Episode: {episode}", True, WHITE)
    screen.blit(episode_text, (10, 10))  # 좌측 상단에 텍스트 표시


    pygame.display.flip()
    

#게임 환경 및 학습 설정
env = rainforce_train_2.PongEnv()
agent = rainforce_train_2.DQNAgent(input_dim = 3, output_dim = 3)
num_episodes = 3000
target_update_freq = 10
batch_size = 64
model_path = "model/dqn_model.pth"

#pygame 초기화
pygame.init()
screen = pygame.display.set_mode((env.width, env.height))
pygame.display.set_caption("DQN Ping-Pong Training")
myFont = pygame.font.SysFont(None, 50)
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train(batch_size)
        total_reward += reward
        state = next_state

        #render(env, screen, episode)

        if done:
            break
    

    if episode % target_update_freq == 0:
        agent.update_target_network()

    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward : {total_reward}, Epsilon : {agent.epsilon}")
        
        torch.save(agent.q_network.state_dict(), model_path)

        

    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

pygame.quit()

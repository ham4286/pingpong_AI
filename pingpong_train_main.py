import pygame
import rainforce_train_2

#게임 화면 크기 및 색상 설정
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

#화면 업데이트 함수 정의
def render(env, screen):
    screen.fill(BLACK)

    pygame.draw.circle(screen, WHITE, (int(env.ball_x), int(env.ball_y)), env.ball_radius)
    pygame.draw.rect(screen, WHITE, (env.paddle_x, env.height - env.paddle_height, env.paddle_width, env.paddle_height))
    pygame.display.flip()

#게임 환경 및 학습 설정
env = rainforce_train_2.PongEnv()
agent = rainforce_train_2.DQNAgent(input_dim = 3, output_dim = 3)
num_episodes = 500
target_update_freq = 10
batch_size = 64

#pygame 초기화
pygame.init()
screen = pygame.display.set_mode((env.width, env.height))
pygame.display.set_caption("DQN Ping-Pong Training")

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

        render(env, screen)

        if done:
            break


    if episode % target_update_freq == 0:
        agent.update_target_network()

    
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward : {total_reward}, Epsilon : {agent.epsilon}")

    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

pygame.quit()

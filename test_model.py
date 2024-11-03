import pygame
import torch
import rainforce_train_2

# 게임 화면 크기 및 색상 설정
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 화면 업데이트 함수 정의
def render(env, screen, score):
    screen.fill(BLACK)
    pygame.draw.circle(screen, WHITE, (int(env.ball_x), int(env.ball_y)), env.ball_radius)
    pygame.draw.rect(screen, WHITE, (env.paddle_x, 0, env.paddle_width, env.paddle_height))  # 패들이 위쪽에 위치
    
    # 스코어 텍스트 렌더링
    font = pygame.font.Font(None, 36)
    score_text = font.render(f'Score: {score}', True, WHITE)
    screen.blit(score_text, (10, 10))  # 화면 좌상단에 스코어 표시

    pygame.display.flip()

# DQN 에이전트 설정 (모델 로드)
class DQNAgent:
    def __init__(self, input_dim, output_dim, model_path):
        self.q_network = rainforce_train_2.DQN(input_dim, output_dim)
        self.q_network.load_state_dict(torch.load(model_path))
        self.q_network.eval()  # 평가 모드로 설정

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():  # 평가 시에는 gradient 계산 필요 없음
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

# 게임 환경 초기화
env = rainforce_train_2.PongEnv()
agent = DQNAgent(input_dim=3, output_dim=3, model_path='model/dqn_model_2000.pth')  # 학습된 모델 경로
num_episodes = 100  # 평가할 에피소드 수

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((env.width, env.height))
pygame.display.set_caption("DQN Ping-Pong Evaluation")
clock = pygame.time.Clock()  # FPS 조절을 위한 Clock 객체 생성
fps = 144  # 초당 프레임 수 설정

# 에피소드 평가
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    score = 0  # 스코어 초기화
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        if reward == 1:
            score += 1  # 점수 증가
    


        
        state = next_state

        render(env, screen, score)  # 스코어를 렌더링 함수에 전달

        clock.tick(fps)  # FPS에 맞춰 루프 속도 조절

    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Score: {score}")

pygame.quit()

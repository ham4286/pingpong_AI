import pygame
import rainforce_train_2

# 게임 화면 크기 및 색상 설정
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 화면 업데이트 함수 정의
def render(env, screen, score):
    screen.fill(BLACK)

    pygame.draw.circle(screen, WHITE, (int(env.ball_x), int(env.ball_y)), env.ball_radius)
    pygame.draw.rect(screen, WHITE, (env.paddle_x, env.height - env.paddle_height, env.paddle_width, env.paddle_height))
    
    # 점수판 표시
    score_text = myFont.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))  # 좌측 상단에 텍스트 표시

    pygame.display.flip()

# 게임 환경 설정
env = rainforce_train_2.PongEnv()
num_episodes = 500

# pygame 초기화
pygame.init()
screen = pygame.display.set_mode((env.width, env.height))
pygame.display.set_caption("Pong Game")
myFont = pygame.font.SysFont(None, 36)

player_score = 0  # 점수 초기화

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # 사용자 입력으로 패들 조정
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  # 왼쪽 화살표 키
            env.paddle_x -= 5
        if keys[pygame.K_RIGHT]:  # 오른쪽 화살표 키
            env.paddle_x += 5

        env.paddle_x = max(0, min(env.paddle_x, env.width - env.paddle_width))  # 패들이 화면 밖으로 나가지 않게 제한

        # 공 움직임
        next_state, reward, done = env.step(0)  # DQN 대신 기본 행동을 사용
        total_reward += reward
        
        render(env, screen, player_score)  # 화면 업데이트

        if done:
            if reward == -1:  # 공을 놓쳤을 경우
                player_score -= 1  # 점수 감소
            else:
                player_score += 1  # 점수 증가
            break

pygame.quit()

import pandas as pd
import numpy as np

csv = pd.read_csv("score.csv", sep = ",",encoding='utf-8-sig')
df = pd.DataFrame(csv)

data = []
#print(csv)
while True:
    # 사용자에게 입력 받기
    이름 = input("이름을 입력하세요 (종료하려면 '종료' 입력): ")
    if 이름 == '종료':
        break
    승패 = int(input("승패를 입력하세요 (1: 승, 0: 패): "))
    시간 = int(input("시간을 입력하세요: "))
    
    # 새로운 데이터 생성g
    new_data = pd.DataFrame({'이름': [이름], '승패': [승패], '시간': [시간]})
    
    # 기존 데이터프레임에 새로운 데이터 추가
    df = pd.concat([df, new_data], ignore_index=True)

# 변경된 데이터프레임을 CSV 파일로 저장
df = df.sort_values(by='시간', ascending=False)

df.to_csv('score.csv', index=False)

print("데이터가 성공적으로 저장되었습니다.")

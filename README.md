# SO-ARM101 공 6개 옮기기 · Colab 강화학습

MuJoCo/MJWarp에서 SO-ARM101이 바구니의 공 6개를 고정된 2×3 격자 칸으로 옮기는 실습입니다.
Colab 노트북은 이 저장소를 `git clone`하고, 분리된 Python 파일을 불러옵니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjeong3150/so101-rl-pjt/blob/main/notebooks/SO101_RL_Colab.ipynb)

저장소: [jjeong3150/so101-rl-pjt](https://github.com/jjeong3150/so101-rl-pjt)

**이 버전은 실행·구문·물리·학습·GPU·뷰어 연결 검증을 수행하지 않았습니다.**
학습 성공률, 소요 시간, T4/L4 메모리 사용량은 측정하지 않았습니다.

## Colab에서 시작하기

1. 위의 **Open In Colab** 버튼으로 `notebooks/SO101_RL_Colab.ipynb`를 엽니다.
2. Colab 런타임을 T4 또는 L4 GPU로 선택합니다.
3. `REPO_URL`의 기본값인 `https://github.com/jjeong3150/so101-rl-pjt`를 사용합니다.
4. 특정 브랜치·태그·커밋을 쓰려면 `GITHUB_REF`를 지정합니다.
5. 셀을 순서대로 실행해 코드를 내려받고, 패키지와 로봇 모델을 설치한 뒤 학습과 뷰어를 엽니다.

저장소만 직접 내려받으려면 다음 명령을 사용합니다.

```bash
git clone https://github.com/jjeong3150/so101-rl-pjt.git
```

공개 저장소는 별도 인증 없이 내려받을 수 있습니다.
비공개 저장소는 본인의 Colab 환경에서 인증을 따로 준비해야 합니다.
노트북 입력란이나 저장소 주소에 토큰을 넣어 저장하지 마세요.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `notebooks/SO101_RL_Colab.ipynb` | 저장소 다운로드, 설정, 학습·뷰어 실행 |
| `environment.py` | 로봇·공·바구니·격자 물리, 관측·행동·보상·종료 조건 |
| `learning.py` | PPO 정책, 경험 수집, 업데이트, 정책 저장·이어 학습 |
| `viewer.py` | 별도 CPU 평가 환경을 보여주는 브라우저 3D 화면 |
| `assets_manifest.json` | 원본 로봇 모델 커밋과 다운로드 파일 목록 |
| `requirements-colab.txt` | Colab용 패키지 목록 |

학습 코드를 수정할 때는 해당 Python 파일을 편집하고 저장소에 반영합니다.
Colab에서는 저장소를 다시 내려받은 뒤 모듈을 불러옵니다.

## 무엇을 학습하나요?

정책이 팔의 다섯 관절과 집게 관절의 목표 각도 변화량을 직접 선택합니다.
미리 작성한 접근·집기·운반 경로, 시간표에 따른 집게 동작, 경로에 더하는 보정은 사용하지 않습니다.
MuJoCo의 위치 액추에이터가 정책이 선택한 목표 각도를 물리적으로 실행합니다.

관측에는 관절 상태, 손끝 위치, 공 위치·속도, 목표 위치와 채움 상태가 들어갑니다.
시뮬레이터의 수치 상태를 사용하며 영상 인식은 포함하지 않습니다.
다음 목표 공과 칸은 과제가 알려주고, 그곳으로 움직이는 행동은 정책이 선택합니다.

공 6개는 바구니 안의 정해진 여섯 위치에서 시작합니다.
먼저 옮긴 공도 계속 물리 계산에 참여하며, 공 하나를 옮길 때마다 초기화하지 않습니다.
실제로 집어 들어 놓은 공들이 대응하는 여섯 칸에 안정적으로 남아 있어야 전체 성공입니다.
이미 놓은 공이 밀려나면 현재 채움 수가 감소합니다.

## 학습 규모와 저장

기본 설정은 T4에서 환경 8개, L4에서 환경 16개이며 노트북에서 변경할 수 있습니다.
상태 관측과 작은 정책 신경망을 사용합니다. 초기 정책이 공을 집지 못할 수 있습니다.
학습 보상, 채운 칸 수, 전체 성공률과 최적화 손실을 구분해서 확인하세요.

정책과 최적화 정보를 저장한 `latest.pt`로 추가 학습할 수 있습니다.
이어 학습할 때 시뮬레이터 상태, 진행 중이던 경험 수집과 난수 흐름은 새로 시작합니다.
Colab 세션이 끝나기 전에 결과를 다운로드하세요.

## 웹 뷰어

GPU 학습과 별개인 CPU 환경 한 대에서 학습 시작 시 정책, 최신 정책, 무작위 행동을 비교합니다.
카메라 회전·확대, 일시정지와 전체 초기화를 사용할 수 있습니다.
정책 가중치는 전체 에피소드 시작 때 반영합니다.

Colab 내부 화면과 본인 세션의 별도 탭 링크를 제공합니다.
별도 탭이 브라우저 보안 설정 때문에 연결되지 않으면 내부 화면을 사용하세요.
런타임과 노트북을 열어 두어야 하며, 공개 터널은 만들지 않습니다.

## 원본 모델과 참고

SO-ARM101 모델은 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)에서 내려받습니다.
원본 모델의 Apache-2.0 라이선스는 다운로드한 `robotstudio_so101/LICENSE`에 있습니다.
로봇 메시 파일은 이 저장소에 포함하지 않으며, 고정한 원본 커밋에서 내려받습니다.

[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) · [NVIDIA Warp](https://github.com/NVIDIA/warp) ·
[mjviser](https://github.com/mujocolab/mjviser) · [Viser](https://github.com/viser-project/viser) ·
[PPO 논문](https://arxiv.org/abs/1707.06347)

이 실습은 시뮬레이션용이며 실제 로봇을 연결하거나 움직이지 않습니다.

# 실험 환경 스펙 및 OS 정리

논문 실험 및 Aerial Gym 실행에 사용하는 **개발 머신 사양**과 **운영 체제·환경**을 정리한 문서입니다.  
재현성·트러블슈팅·보고 시 참고용으로 사용하세요.

---

## 1. PC 스펙 (하드웨어)

| 항목 | 사양 |
|------|------|
| **CPU** | Intel Core i9-13900KF |
| **GPU** | NVIDIA GeForce RTX 3090 |
| **RAM** | 64 GB |
| **기타** | 듀얼 부팅 (Windows + Ubuntu) |

---

## 2. 운영 체제 및 부팅 방식

| 항목 | 내용 |
|------|------|
| **실험용 OS** | **Ubuntu 22.04 LTS** (네이티브 설치) |
| **부팅 방식** | **듀얼 부팅** (Windows와 Ubuntu 중 선택 부팅) |
| **비고** | WSL2가 아닌 **Ubuntu 네이티브**에서 Aerial Gym / Isaac Gym 실행. GPU·드라이버를 직접 사용하므로 NVIDIA 드라이버는 Ubuntu에 설치된 버전을 사용. |

---

## 3. 소프트웨어 환경 (참고)

| 항목 | 내용 |
|------|------|
| **Conda 환경 이름** | **aerialgym** |
| **Python 버전** | **3.8** |
| **Isaac Gym** | NVIDIA Isaac Gym 설치 후 Aerial Gym에서 사용 |
| **CUDA / 드라이버** | Ubuntu 22.04에 설치한 NVIDIA 드라이버·CUDA에 따름. (`nvidia-smi`, `nvcc --version`으로 확인 후 필요 시 아래 예시 칸에 기록.) |

활성화 예: `conda activate aerialgym`

---

## 4. 프로젝트 경로 (본 환경 기준)

- **워크스페이스 루트**: `/home/user/aerial_gym_simulator`
- **환경 루트**: `/home/user/aerial_gym_simulator/environment`
- **Runner 실행 디렉터리**: `/home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games`

---

## 5. 문서 업데이트

- GPU 드라이버 버전, CUDA 버전, Conda 환경 이름, Isaac Gym 경로 등을 나중에 확인했으면 위 표나 아래에 추가해 두면 재현·디버깅에 유리합니다.

```text
# 예시 (확인 후 채우기)
# NVIDIA Driver: 
# CUDA: 
# Conda env name: aerialgym  (Python 3.8)
# Isaac Gym path: 
```

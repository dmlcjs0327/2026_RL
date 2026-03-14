# SHVD Workspace

이 저장소는 SHVD 연구를 위해 최상위에서 `paper/`와 `environment/`로 역할을 분리해 관리합니다.  
논문 작성 자산과 시뮬레이션/실험 자산이 섞이면서 동선이 복잡해지는 문제를 줄이기 위해 구조를 재정리했습니다.

## 현재 구조 개요

- `paper/`: 논문 원고, 참고문헌, 논문 설계 문서
- `environment/`: Aerial Gym 기반 시뮬레이터, 강화학습 코드, 실험 스크립트, 환경 문서
- `.vscode/`: Cursor/VS Code 작업 설정. LaTeX Workshop와 Tectonic 기반 PDF 빌드 설정 포함
- `.gitignore`: 학습 산출물, 체크포인트, LaTeX 빌드 산출물 무시 규칙
- `LICENSE`: 저장소 라이선스

## Top-Level Directory Guide

### `paper/`

논문 작성 관련 자산을 모아 둔 작업 공간입니다.

- `template.tex`: 현재 메인 원고 템플릿
- `Definitions/`: MDPI 클래스 및 스타일 파일
- `references.bib`: 원고에서 사용할 참고문헌 데이터
- `reference_papers/`: 수집한 참고 논문과 BibTeX 정리 파일
- `docs/`: 논문 실험 설계, 베이스라인 정의, 논문 컨텍스트 메모
- `.latex-build/`: LaTeX Workshop/Tectonic 빌드 산출물

### `environment/`

실제 드론 시뮬레이션, 학습, 평가를 수행하는 코드베이스입니다.

- `aerial_gym/`: 시뮬레이터 본체와 학습 코드
- `resources/`: 로봇 URDF, 환경 에셋, 메쉬 등 정적 리소스
- `scripts/`: 실행 보조 스크립트와 테스트 스크립트
- `runs/`: 학습 로그, TensorBoard 출력, 체크포인트 저장 위치
- `docs/`: 실행 환경, 터미널 명령, 분석 메모, 원본 레퍼런스 문서
- `setup.py`, `pyproject.toml`, `requirements.txt`: 패키징 및 의존성 정의
- `mkdocs.yml`: 문서 빌드 설정

## `environment/aerial_gym/` 내부 구성

핵심 시뮬레이터/학습 로직은 `environment/aerial_gym/` 아래에 있습니다.

- `rl_training/`: PPO(`rl_games`) 및 off-policy 학습 코드와 설정 파일
- `task/`: position setpoint, goal-conditioned task 등 태스크 정의
- `config/`: task, robot, sensor, controller, environment 관련 설정
- `env_manager/`: 시뮬레이션 환경 및 에셋 관리
- `robots/`: 기체/로봇 모델 추상화
- `sensors/`: 카메라, LiDAR, IMU 등 센서 관련 코드
- `sim/`: 시뮬레이터 빌더와 실행 구성
- `sim2real/`: sim-to-real 관련 자산과 추론/변환 코드
- `nn/`, `utils/`, `assets/`: 보조 신경망 모듈, 유틸리티, 에셋 로딩 코드

## 문서 위치

### 논문 문서

- 논문 작업 시작: `paper/template.tex`
- 논문 컨텍스트 메모: `paper/docs/drones_paper_context.md`
- 실험 설계 문서: `paper/docs/experiments/`
- 참고 논문: `paper/reference_papers/`

### 환경 문서

- 환경 진입점: `environment/README.md`
- 실행 명령 모음: `environment/docs/setup/terminal_commands.md`
- 개발/실험 환경 스펙: `environment/docs/setup/environment_spec.md`
- 분석 및 테스트 기록: `environment/docs/analysis/`
- 원본 Aerial Gym 참고 문서: `environment/docs/reference/`

## 작업 흐름 권장

### 논문 작업

1. `paper/template.tex`에서 원고 작성
2. `paper/docs/experiments/`에서 실험 설계와 표/그림 구조 정리
3. `paper/reference_papers/`와 `references.bib`로 참고문헌 관리

### 실험 작업

1. `environment/docs/setup/terminal_commands.md`에서 실행 명령 확인
2. `environment/aerial_gym/rl_training/`에서 학습 설정 수정
3. 결과는 `environment/runs/`에서 확인

## 빠른 시작

```bash
cd /home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games
python runner.py --task=position_setpoint_task --num_envs=256 --headless=True --use_warp=True
```

논문 PDF 빌드는 `paper/template.tex`를 저장하면 Cursor의 LaTeX Workshop 설정에 따라 자동으로 수행됩니다.

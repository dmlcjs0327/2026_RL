# Pretrain-then-Adapt for UAV Reinforcement Learning

**A Systematic Comparison of Full Fine-Tuning, Input-Branch-Frozen, and LoRA Adaptation Strategies in Aerial Gym**

---

| | |
|---|---|
| **Target venue** | *Drones* (MDPI, IF 4.8, Q1 Remote Sensing) |
| **Format** | Full research article, ~18–22 pages |
| **Simulator** | Aerial Gym Simulator v2.0.0 (Warp backend) |
| **RL library** | Sample Factory |
| **Hardware** | RTX 3090 / i9-13900KF / 64GB RAM |
| **Timeline** | 4 months (post thesis submission) |

---

## 1. Topic and Significance

### 1.1 Topic

본 연구는 자율 무인 항공기(UAV)의 학습 기반 항법을 위한 강화학습(RL) 정책 훈련에서, **기반 정책을 한 번 사전학습한 후 다운스트림 비행 과업에는 효율적 적응 전략으로 소량 조정**하는 패러다임의 유효성과 세 가지 적응 전략(Full Fine-Tuning, Input-Branch-Frozen, LoRA) 간 trade-off를 Aerial Gym 시뮬레이션에서 체계적으로 분석한다.

### 1.2 Practical Significance for Autonomous Flight

학습 기반 UAV 항법 정책의 실제 배포에는 다음과 같은 만성적 병목이 존재한다.

**(i) 매 임무마다 처음부터 재학습.** 현재 UAV RL 연구의 관행은 새 임무·환경마다 무작위 초기화에서 시작해 수천만~수억 환경 상호작용을 소비한다. 실무자가 새 검사 임무, 새 비행장, 새 장애물 분포에 대응하려면 매번 전체 학습 사이클을 반복해야 한다.

**(ii) 직접 학습의 본질적 난이도.** 현재 관행은 종종 "불안정한 비선형 동역학 제어 + 장애물 회피 + 목표 도달 + (경우에 따라) 동적 객체 대응"을 **무작위 초기화에서 한 번에 학습**시키려 한다. 사람이 언어를 배울 때 바로 긴 문장부터 시작하지 않고 단어·발음부터 익숙해진 뒤 문장으로 넘어가는 것이 효과적이듯, 이처럼 복합적인 능력이 요구되는 과업을 처음부터 통합 학습시키는 것은 탐색 공간이 과도하게 넓어 수렴이 불안정해진다. Scratch PPO 학습이 seed·hyperparameter에 민감한 이유가 여기에 있다.

**(iii) 엔지니어링 공수의 비효율적 분배.** 자율비행 팀의 RL 엔지니어 시간 상당 부분이 hyperparameter tuning과 학습 안정화에 소진된다. 이 공수는 매 새 과업마다 반복된다.

### 1.3 A Verified Paradigm from NLP/CV — And Why It Transfers

NLP와 CV는 이 병목을 **사전학습 + 효율적 적응**으로 해결했다. GPT, LLaMA 계열은 대규모 텍스트로 사전학습된 기반 모델을 한 번 확보한 후, 다운스트림 과업에는 Full Fine-Tuning, Adapter, **LoRA (Low-Rank Adaptation)** 같은 전략으로 소량만 조정한다 (Hu et al. 2021). 이 패러다임이 분야 생산성을 수십 배 끌어올렸다.

이 접근이 효과적인 것은 **단순한 관행이 아니라 학습 원리상 합리적**이기 때문이다. 언어 학습자가 긴 문장을 바로 구성하기보다 단어·발음을 먼저 익히는 것처럼, 사전학습은 "일반적 패턴"을 먼저 습득시키고 다운스트림 적응 단계에서는 **이미 확보된 표현 위에 과업 특화 조정만 수행**한다. 탐색 공간이 좁아져 수렴이 안정화되고 샘플 효율이 개선된다.

**자율비행에 이 논리가 그대로 적용된다**. UAV에 "동적 장애물을 회피하며 효과적으로 기동하는 능력"을 한 번에 학습시키는 대신, 먼저 **기본 비행 제어·공간 인식·goal-seeking** 같은 일반적 비행 능력을 사전학습으로 확보하고, 이후 구체적 임무(조밀 장애물 회피, 좁은 복도 통과, 정밀 도킹, 동적 회피)에 대한 적응은 소규모 조정으로 처리할 수 있다. 동일한 총 학습 자원을 사전학습과 적응에 분배하는 것이, 전부를 다운스트림 직접 학습에 투입하는 것보다 원리적으로 더 효과적일 가설은 합리적이다.

### 1.4 Explicit Gap in UAV RL

이 패러다임을 UAV RL에 이식한 연구는 단편적으로만 존재한다.

- **Kulkarni & Alexis (2024, ICRA, "Deep Collision Encoding")**: Aerial Gym에서 depth encoder 사전학습 후 동결, 위에 RL 정책. **지각 인코더 동결 단일 전략** (본 연구의 Input-Branch-Frozen에 해당).
- **SLowRL (2026, arXiv 2603.17092)**: Unitree Go2에서 sim2real LoRA. **Quadruped, 단일 task**.
- **CORAL (2026, arXiv 2603.09298)**: VLA 모델에서 per-task LoRA. **Manipulation 도메인**.

**누구도 하지 않은 것**: UAV 항법 도메인에서 Full Fine-Tuning, Input-Branch-Frozen, LoRA 세 전략을 **총 환경 상호작용 예산을 일치시킨 공정 조건에서 직접 비교**하고, 각 전략의 trade-off를 다양한 다운스트림 과업 특성에 걸쳐 체계적으로 매핑한 연구.

### 1.5 What This Work Delivers to Practitioners

- 새 임무 적응 시 full retraining을 회피할 수 있는 검증된 경로
- 자기 과업 특성에 맞는 적응 전략 선택의 과학적 근거
- **Jetson Orin Nano급 온보드 엣지 프로세서에 탑재 가능한 소형 정책 크기(~280K params, FP16 ~0.56 MB) 내에서 효율적 적응 가능성 입증** — LLM 스케일이 아닌 "배포 현실"에 맞는 영역에서도 pretrain-adapt 패러다임이 작동하는가에 대한 답
- Aerial Gym 기반 재현 가능한 오픈소스 protocol

---

## 2. Problem Formulation

### 2.1 Setting

Aerial Gym 시뮬레이터에서 관측·행동 공간을 공유하는 UAV 항법 과업군 $\{\mathcal{T}_i\}$를 상정한다. 관측·행동·컨트롤러 조합은 Aerial Gym의 **공식 `navigation_task` 경로**를 그대로 채택하여, 기존 코드 인프라와의 정합성을 최대화한다 (`task/navigation_task/navigation_task.py`).

- **관측 공간** $\mathcal{O}$ (81차원): `navigation_task_config.py` default `observation_space_dim = 17 + 64 = 81`와 일치. 실제 index layout:
    - `[0:3]` unit vector to goal
    - `[3]` distance to goal (scalar)
    - `[4:7]` euler angles (roll, pitch, yaw)
    - `[7:10]` linear velocity
    - `[10:13]` angular velocity
    - `[13:17]` previous 4-d action (internally stored; policy는 3-d velocity 출력)
    - `[17:81]` VAE latent (depth camera 270×480 이미지를 사전학습된 $\beta$-VAE로 64-d로 압축)
- **행동 공간** $\mathcal{A}$ (3차원 continuous): velocity command ($v_x, v_y, v_z$), Aerial Gym의 `lmf2_velocity_control` 컨트롤러가 이를 body rates 및 motor RPM으로 변환. 100 Hz control loop.
- **기체**: Aerial Gym registry의 `lmf2` (카메라 내장 planar quadrotor, `navigation_task_config.py` default).

**관측 설계 노트**: 64-d VAE latent는 ntnu-arl(Kulkarni group, Aerial Gym v2.0.0 배포 포함)이 공개한 사전학습 체크포인트(`ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth`)이다. 이 부분은 본 연구의 적응 대상이 **아니며**, 모든 비교 조건에서 동일하게 frozen 상태로 유지된다. 본 연구의 사전학습-적응 패러다임은 VAE **위의 policy 네트워크**만을 대상으로 한다. 이 설계는 공정 비교의 혼란 변인을 최소화하고 DCE (Kulkarni & Alexis 2024)와의 직접 비교 가능성을 확보한다.

### 2.2 Base Policy (Jetson-Scale Onboard Budget)

본 연구의 핵심 제약은 **Jetson Orin Nano급 온보드 컴퓨터에서 실시간 추론 가능한 정책**을 전제한다는 점이다. 이 제약은 다음 두 가지 실무적 이유에서 중요하다.

1. **배포 가능성**: 학습된 정책이 실제 UAV 온보드에 탑재 가능해야 자율비행 연구로서 가치가 있다. Jetson Orin Nano는 ~40 TOPS (INT8), 10 TFLOPS (FP16), 5.2 GB available VRAM으로 소형 UAV에 널리 사용되는 엣지 플랫폼.
2. **적응 비용 효과**: 정책이 작을수록 Full-FT 대비 LoRA의 parameter efficiency 우위가 **상대적으로 작아질** 수 있다. 본 연구는 그럼에도 불구하고 이 소형 영역에서 각 전략의 trade-off를 정확히 매핑한다. 이는 LLM처럼 거대한 모델에서 LoRA의 당연한 우위가 관찰되는 것과 달리, **제약된 크기의 UAV 정책에서도 의미 있는 trade-off가 존재하는지**를 검증하는 가치가 있다.

이에 따라 base policy $\pi_{\theta_0}$는 **총 ~280K 파라미터 규모의 recurrent 정책** (Section 4.2). FP32 기준 ~1.1 MB, FP16 quantization 시 ~0.56 MB로 Jetson Orin Nano에 여유롭게 탑재 가능. 280K/FP16 환경에서 추론은 **100 Hz 이상 여유**이며 (FP16 10 TFLOPS × 작은 모델 size로 compute-bound 여부조차 문제되지 않음), memory bandwidth 68 GB/s 병목은 LLM 사례에서만 유의미하고 소형 recurrent policy에는 무관.

**실제 플랫폼 정합성 (중요)**: 본 연구의 "Jetson-scale + X500" motivation은 가정이 아니라 **상업적으로 확립된 플랫폼**에 정확히 대응한다. 구체적으로:
- **Holybro X500 v2 + Pixhawk Jetson Baseboard** (Jetson Orin NX/Nano 내장 + Pixhawk 6X 통합)는 PX4 공식 Dev Kit로 시판 중
- **Aerial Gym v2의 공식 Sim2Real 가이드**(NTNU 배포)는 **Holybro X500 v2를 기준 플랫폼으로 명시적 채택**하며 PX4 experimental 모듈까지 제공
- 본 연구의 ~280K 정책은 이 경로에서 별도 모델 압축 없이 배포 가능

### 2.3 Three Adaptation Strategies

다운스트림 과업 $\mathcal{T}_i^{\text{down}}$에 대해 비교하는 세 전략. **모든 조건에서 외부 VAE(64-d latent 생성)는 공통으로 frozen**이며, 아래 "Frozen" 열은 policy 네트워크 **내부**의 추가 frozen 영역을 의미한다.

| 전략 | Policy 내부 Frozen (VAE 외에 추가) | Trainable Region | Trainable Params | Inductive Bias |
|------|-----------------------------------|------------------|------------------|----------------|
| **Full-FT** | — | Policy 전체 | ~280K | 최대 유연성, prior overwrite 위험 |
| **Input-Branch-Frozen** | Input MLP (proprio-17 branch + VAE projection) | Fusion + GRU + heads | ~150K | 지각 표현 보존, 정책만 특화 |
| **LoRA** | 전체 policy backbone | Linear layer의 low-rank update | ~7K @ r=8 | 저차 제약을 통한 prior 보존 |

**이전 초안에서 이름 변경**: 이전 초안은 "Encoder-Frozen"으로 부르다 "어느 encoder를 의미하는가"(VAE? policy 내부 input MLP?) 혼동 여지가 있었다. 본 연구에서는 **VAE는 모든 조건에서 외부 frozen 모듈**이고, "Input-Branch-Frozen" 전략은 그 위에 **policy 네트워크 내부의 input-branch MLP까지 추가로 frozen**한다는 의미로 rename.

**중요한 framing**: 세 전략의 trainable parameter magnitude는 서로 다르지만 (최대 40배 차이), 이는 **"파라미터 수를 통제한 비교"가 아니라 "동일한 사전학습된 backbone 위에 서로 다른 inductive bias를 갖는 적응 전략들 간의 공정 비교"**이다. 각 전략은 자신의 inductive bias가 허용하는 최대 표현력을 활용한다. 비교의 공정성은 파라미터 수가 아니라 **(i) 동일한 backbone 초기값, (ii) 동일한 환경 상호작용 예산, (iii) 동일한 PPO hyperparameter, (iv) 동일한 평가 protocol**로 확보된다 (Section 4.8).

LoRA의 default 주입 범위는 **Linear layer에 한정** (fusion MLP 2개 + actor/critic head의 첫 Linear). GRU는 frozen. 근거는 §4.7, GRU LoRA 효과 검증은 ablation A6.

### 2.4 Core Research Question

> **Jetson Orin Nano 급 소형 UAV 항법 정책에서, 총 환경 상호작용 예산 $B_{\text{total}}$을 명시적으로 통제한 공정 비교 하에, 세 적응 전략은 다음 다섯 축에서 각각 어떻게 다른가:**
>
> **(i) 샘플 효율, (ii) 최종 성능, (iii) 수렴 안정성, (iv) 하이퍼파라미터 민감도, (v) 사전학습 능력 보존도. 그리고 이 차이는 다운스트림 과업의 어떤 특성에 따라 변하는가?**

### 2.5 Scope

- **Aerial Gym 시뮬레이션 전용.** 실기체 sim2real은 명시적 future work.
- **회전익 단일 기체.** Aerial Gym `lmf2`. 고정익/VTOL은 Aerial Gym 자체가 미지원.
- **Aerial Gym navigation 표준 경로.** 관측(17-d state + 64-d VAE latent), 행동(3-d velocity command), 컨트롤러(`lmf2_velocity_control`)를 기존 navigation_task와 일치시켜 fair comparison 확보.
- **Jetson Orin Nano 호환 소형 backbone (~280K).** 더 큰 backbone은 future work.
- **PPO only**. 본 연구는 on-policy PPO에 한정한다. Off-policy 알고리즘(SAC, TD3)과의 비교는 **명시적으로 scope 외**이며 future work. 이 축소 결정의 근거는 (i) Sample Factory가 PPO 중심 라이브러리이고 off-policy 지원이 제한적이므로 공정 비교 인프라 확보 부담이 크다, (ii) 알고리즘 간 비교는 본 연구의 축(적응 전략 비교)과 직교하며 혼란 변인을 늘린다, (iii) Wołczyk et al. (2024), SLowRL (2026), CORAL (2026) 등 선행 연구가 모두 on-policy 중심이라 비교 대상 문헌과의 정합성 확보.
- **단일 기체 환경.** 다기체/편대는 Aerial Gym 미지원.

---

## 3. Related Work and Common Limitations

### 3.1 Group A — Scratch RL for UAV Navigation

대부분의 UAV RL 연구는 무작위 초기화에서 PPO 학습을 수행한다.

- **Zhou et al. (2024, arXiv 2412.12442)**: Multi-Task RL for Quadrotors, shared + task-specific encoder, multi-critic. 모든 task scratch 학습.
- **Curriculum RL for Racing (2026, arXiv 2602.24030)**: curriculum + domain randomization.
- **Fan et al. (2025, arXiv 2503.14352)**: Lidar 기반 동적 장애물 회피, end-to-end DRL.
- **Drone Aerobatics MTRL (2026, arXiv 2602.10997)**: 기하학적 대칭성 활용 MTRL 가속.

**공통 한계**: (i) 매 과업마다 처음부터 재학습, (ii) seed 분산·hyperparameter 민감도 분석 부재, (iii) 적응 전략 축 자체가 없음.

### 3.2 Group B — Pretraining + Single Adaptation Strategy

- **Kulkarni & Alexis (2024, ICRA, DCE)**: Depth encoder supervised pretraining 후 **frozen**. Aerial Gym 공식 navigation 예제. **지각 인코더 동결 전략의 전형** (본 연구의 Input-Branch-Frozen 대응).
- **Zehnder et al. (2024, arXiv 2410.15979)**: Differentiable simulation에서 state representation pretraining 후 **full fine-tune**.
- **Becker-Ehmck et al. (2020)**: VAE latent 후 full fine-tune.

**공통 한계**: 각 연구가 단일 적응 전략만 사용. 전략 선택 근거가 실험적 비교가 아닌 저자 편의.

### 3.3 Group C — LoRA in RL

- **SLowRL (2026, arXiv 2603.17092)**: Unitree Go2 sim2real, frozen base + LoRA. **rank-1로도 충분**, actor+critic 동시 적응 필요, 46.5% 시간 단축. **단일 task 단일 embodiment**.
- **LoRASA (2025, arXiv 2502.05573)**: Multi-agent RL, agent-specific LoRA.
- **CORAL (2026, arXiv 2603.09298)**: VLA + per-task LoRA library. Manipulation.
- **arXiv 2603.11653 (2026)**: VLA + on-policy RL + LoRA의 시너지로 catastrophic forgetting 자연 완화.

**공통 한계**: UAV 도메인 부재. LoRA의 존재 가치만 보임, 경쟁 전략과 ceteris paribus 비교 없음.

### 3.4 Group D — RL Fine-Tuning Stability

- **Wołczyk et al. (2024, arXiv 2402.02868)**: RL fine-tuning에서 state coverage gap으로 인한 catastrophic forgetting. Knowledge retention 기법 효과 입증.
- **Dohare et al. (Nature 2024)**: PPO 포함 표준 RL의 plasticity loss 취약성.

이들은 **Full-FT의 위험성**을 시사하며, LoRA·Frozen이 이론적으로 더 안정적일 것이라는 본 연구의 가설 근거.

### 3.5 Consolidated Gap

위 네 그룹을 가로질러 다음 공백이 있다.

> **UAV RL에서 사전학습 후 세 가지 주요 적응 전략(Full-FT / Input-Branch-Frozen / LoRA)을, 동일 기반 정책·동일 다운스트림 환경·동일 총 예산 하에서 공정 비교하고, 각 전략의 trade-off를 다양한 다운스트림 과업 특성(환경 복잡도, 시간 구조, 외란 수준)에 걸쳐 체계적으로 매핑한 연구가 없다.**

본 연구가 이 공백을 겨냥한다.

---

## 4. Proposed Method

### 4.1 Overall Pipeline

**Phase 1 — Pretraining.** 단순한 goal-reaching 환경에서 기반 정책 $\pi_{\theta_0}$를 순수 PPO로 학습 (보조 손실 없음; §4.3 참조).

**Phase 2 — Downstream Adaptation.** 각 다운스트림 과업에 대해 $\theta_0$를 출발점으로 세 적응 전략을 PPO로 적응시킨다. **세 전략 모두 동일한 PPO hyperparameter, 동일한 환경 상호작용 예산, 동일한 평가 protocol 사용.**

### 4.2 Policy Architecture (Jetson-Scale ~280K params)

관측 공간이 **81-d (17-d state + 64-d VAE latent)**, 행동 공간이 **3-d velocity command**인 recurrent 정책. Aerial Gym `navigation_task` 공식 obs layout(§2.1)과 1:1 매핑되도록 설계.

```
Observation (81-d, Aerial Gym navigation_task standard)
├── Proprio+Goal+PrevAction (17-d): navigation_task.py 인덱스 [0:17]
│     unit_vec_to_goal(3) + dist(1) + euler(3) + lin_vel(3)
│     + ang_vel(3) + prev_action(4)
│       └── MLP: 17 → 64 → 64                                        [~6K params]
└── VAE latent (64-d): β-VAE encoder output, 인덱스 [17:81]
    VAE는 외부 frozen 모듈 (모든 조건 공통)
        └── MLP projection: 64 → 64 → 64                             [~8K params]

Concatenate (64 + 64 = 128-d)
└── Fusion MLP: 128 → 128 → 128                                      [~33K params]
    └── GRU: hidden=128, 1 layer                                     [~99K params]
        ├── Actor head: 128 → 64 → 6 (mean+logstd × 3)               [~8K params]
        └── Critic head: 128 → 64 → 1                                [~9K params]

Total policy trainable: ~163K + ~14K input + ~99K GRU = ~280K
Activation: ELU
FP32: ~1.1 MB | FP16 quantized: ~0.56 MB → Jetson Orin Nano 여유 탑재
```

**2개 branch 선택 근거 (Option A)**: Navigation_task의 실제 관측 layout(§2.1)은 goal-관련 4-d + kinematic 9-d + prev_action 4-d가 처음 17-d에 인접 배치되어 있다. 이를 강제로 세 branch로 분해하는 것(예: goal 4 / kinematic 9 / prev_action 4)은 인덱스 경계 처리와 구현 공수만 늘리고 학습 신호 상의 이득은 불명확하다. 본 연구는 **17-d 전체를 단일 proprio MLP로 통합**하고 VAE 64-d와 concat하는 2-branch 구조를 택한다. 이는 navigation_task의 default obs 처리 관행과도 일치하며 구현 부채를 최소화한다.

**GRU 파라미터 정확 계산**: GRU(128, 1-layer) = $3 \times (128 \cdot 128 + 128 \cdot 128 + 128) = 3 \cdot 32,896 = 98,688 \approx 99K$.

**GRU hidden=128 선택 근거** (DCE의 64 대비 2배): DCE의 GRU hidden=64는 하나의 구체적 baseline일 뿐이며, 본 연구의 ~280K 총 예산 내에서 fusion 이후 recurrent 용량을 약간 확대해 **본 연구의 OOD 과업(D3 stationkeeping, D4 dynamic obstacles)에서 belief tracking 여유를 확보**한다. 참고로 같은 생태계의 후속 연구인 Semantically-driven DRL (NTNU, 2025)은 Aerial Gym + Sample Factory APPO에서 **hidden=512**까지 확장해 실기체 검증에 성공했으므로, 128은 범위 내 보수적 선택이다. 학습 불안정성 발생 시 64로 축소하는 것을 M1 smoke test 결정 게이트로 보류. 공정 비교의 축은 "완전히 동일한 하이퍼파라미터"가 아니라 **(i) 관측·행동 인터페이스, (ii) 학습 예산, (iii) 동일 backbone 초기값 하의 적응 전략**이다.

**설계 근거 요약**:
- Aerial Gym `navigation_task` 표준 I/O와 일치 → 기존 env·reward 인프라 재사용, DCE 및 VAE 기반 baseline과 직접 비교 가능.
- GRU 포함 → depth camera 관측의 시간적 통합 (POMDP belief tracking).
- ~280K → **Jetson Orin Nano 추론 가능한 소형 크기**. 이는 본 연구의 핵심 제약이자 motivation (§2.2 참조).
- VAE latent는 외부 frozen 모듈 → 본 연구의 적응 대상에서 제외, 공정 비교의 혼란 변인 최소화.

### 4.3 Pretraining Objective

사전학습은 **순수 PPO**로 진행. 이전 초안의 dynamics prediction 보조 손실 $\mathcal{L}_{\text{dyn}}$은 Sample Factory Learner의 구조적 수정이 필요해 제거 (§4.9.5 구현 부채 논의 참조).

$$\mathcal{L}_{\text{pre}} = \mathcal{L}_{\text{PPO}} = \mathcal{L}_{\text{clip}} + c_v \mathcal{L}_{\text{value}} + c_e \mathcal{L}_{\text{entropy}}$$

표준 PPO objective: clipped surrogate + value loss ($c_v=0.5$) + entropy bonus ($c_e=0.005$). 이 단순화는 구현 부담을 줄이고 framework-level 수정을 회피한다. 보조 손실의 이득은 future work로 보류.

### 4.4 Strategy 1 — Full Fine-Tuning

$\theta_0$를 초기값으로 모든 policy parameter unfrozen (VAE는 본 연구 시작 전부터 frozen). 표준 PPO로 다운스트림 보상 최대화. Trainable: **~280K** (~100% of policy).

### 4.5 Strategy 2 — Input-Branch-Frozen

VAE(외부 모듈)에 더해 **policy 내부의 2개 input MLP 모두 frozen** (17-d proprio branch MLP + 64-d VAE projection MLP), fusion + GRU + heads만 학습. Trainable: **~150K** (~54%). DCE 스타일 적응의 변형.

이 명명은 이전 초안 "Encoder-Frozen"에서 rename한 것이다. "Encoder"가 VAE를 지칭하는지 policy 내부 input MLP를 지칭하는지 혼동 여지가 있어 명확화. 본 전략에서는 외부 VAE(모든 조건 공통 frozen) **위에 추가로** policy 내부 input-branch MLP들까지 frozen 한다.

### 4.6 Strategy 3 — LoRA (Linear-only default)

Policy backbone 전체 frozen 후 다음 Linear layer에만 rank-$r$ LoRA 주입:

- Fusion MLP의 2개 Linear ($128 \times 128$, $128 \times 128$)
- Actor head 첫 Linear ($128 \times 64$)
- Critic head 첫 Linear ($128 \times 64$)
- **GRU: frozen (LoRA 없음)** — 근거는 §4.7

**LoRA 수식**: 각 frozen weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$에 대해
$$W' = W + \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, A \in \mathbb{R}^{r \times d_{\text{in}}}$$

- $A$: Kaiming uniform 초기화, $B$: zero 초기화 → 적응 시작 시 $\Delta W = 0$
- $\alpha/r = 2$, default $r = 8$
- **Actor와 Critic 모두 LoRA** (SLowRL 권고: actor-only 시 value mismatch)

**Trainable params (r=8)**:
- Fusion LoRA 1: $(128 \times 8) + (8 \times 128) = 2,048$
- Fusion LoRA 2: $(128 \times 8) + (8 \times 128) = 2,048$
- Actor head LoRA: $(128 \times 8) + (8 \times 64) = 1,536$
- Critic head LoRA: $(128 \times 8) + (8 \times 64) = 1,536$
- **합계: ~7,168 (~2.6% of policy)**

### 4.7 GRU Adaptation as Ablation

**GRU를 default LoRA 대상에서 제외하는 근거**:

1. **구현 표준성**: PyTorch `nn.GRU`는 cudnn packed weights를 사용해 PEFT 라이브러리의 표준 지원 대상이 아니다. Linear-only는 custom LoRALinear로 비교적 쉽게 구현.
2. **이론적 합리성 (제한적 적용)**: 본 연구의 다운스트림 과업 중 **D1, D2는 사전학습과 동일한 "goal-reach + move on" 시간 구조**를 공유하며, 이 범위에서는 GRU의 belief tracking 기능이 재사용 가능하다. **D3 (sustained hover) · D4 (dynamic obstacles)는 시간 구조가 다르지만**, 이 범위에서 GRU 적응이 실제로 필요한지는 **Ablation A6**가 empirical로 직접 검증한다 (§4.9의 A6 경로). 즉 default는 "GRU 동결이 D1/D2에 충분하다는 가설", A6는 "D3/D4에서 GRU 동결이 bottleneck인가"의 검증.
3. **RL에서 GRU의 학습 불안정성**: GRU는 BPTT 불안정성, gradient vanishing/exploding으로 RL에서 다루기 까다로운 모듈. LoRA 저차 제약이 gate dynamics를 왜곡할 가능성.
4. **Parameter efficiency 주장 강화**: Linear-only LoRA가 ~2.6% trainable로 Full-FT 성능의 상당 부분을 회복하면 강한 결론.

**GRU 적응의 효과는 Ablation A6에서 직접 검증** — **시간 구조 변화가 큰 D3, D4에 한정해** LoRA+GRU 조건을 측정해 marginal contribution 정량화. 근거 2의 가설이 D1/D2에서 성립하고 D3/D4에서 깨지는 양상이 관찰되면, 이는 논문의 "task characteristics에 따른 전략 선택" contribution 3를 직접 뒷받침.

**공정 비교 framing**: §2.3에서 논의한 바와 같이, 본 연구의 공정성은 "정확히 동일한 파라미터를 학습"이 아니라 "동일한 사전학습 backbone, 동일한 환경 상호작용 예산, 동일한 PPO hyperparameter 하의 전략별 inductive bias 비교"이다. A6 ablation이 "LoRA가 GRU 동결 때문에 불이익을 받는가"라는 reviewer 질문에 empirical로 답한다.

### 4.8 Fair Comparison Protocol

**총 환경 상호작용 예산** $B_{\text{total}} = B_{\text{pre}} + B_{\text{down}}$**을 통제.** 이는 본 논문이 RL 적응 비교 문헌에서 자주 누락되는 critical confound를 정면으로 다루는 부분이다.

| 조건 | $B_{\text{pre}}$ | $B_{\text{down}}$ | 총 $B_{\text{total}}$ |
|------|------------------|-------------------|----------------------|
| Scratch | 0 | $2 \times 10^8$ | $2 \times 10^8$ |
| Pre + (Full-FT / Frozen / LoRA) | $5 \times 10^7$ | $1.5 \times 10^8$ | $2 \times 10^8$ |

구체 수치: $B_{\text{pre}}^* = 5 \times 10^7$ env steps (사전학습 plateau), 다운스트림 $1.5 \times 10^8$ env steps.

**Budget 비대칭에 대한 명시적 방어**: Scratch는 다운스트림에 $2 \times 10^8$ 스텝을, 사전학습 기반 조건들은 $1.5 \times 10^8$ 스텝을 받는다. 즉 Scratch는 **다운스트림 학습 시간에서 33% 더 많은 양을 받는** 구조다. 이는 의도적 설계다:

- **"pretrain-adapt이 공짜가 아니다"**라는 사실을 수치로 반영. 사전학습 자체가 $5 \times 10^7$ 스텝을 **소모**하므로, 이를 보상하지 않으면 "pretrained가 유리한 건 당연"이라는 reviewer 비판이 합당하다.
- Pretrained 조건이 "다운스트림 학습량이 33% 적은" 상태에서도 Scratch를 따라잡거나 추월한다면, 그 이득이 **진짜 transfer에서 온 것**임을 강하게 입증한다.
- 이 비대칭은 현재 LLM·VLA 분야의 pretrain-adapt 비교 관행과 일치 (Wołczyk et al. 2024; CORAL 2026). 본 연구는 이 관행을 UAV RL에 **명시적으로** 적용한 첫 연구가 된다.

**왜 이 통제가 중요한가**: 사전학습이 "공짜 데이터"가 아니라 명시적 비용이라는 점을 인정해야 fair comparison이 성립한다.

### 4.9 Implementation Specification (작업 목록 기반)

이 섹션은 Aerial Gym v2 코드베이스에 대한 구체적 검토를 바탕으로, 본 연구가 **실제로 무엇을 구현해야 하는지**를 정직하게 나열한다. 이전 초안은 "Aerial Gym 기성 도구를 그대로 끼우면 된다"는 톤이었으나, 코드 레벨에서 확인한 결과 다수 항목이 신규 구현 또는 수정이 필요했다. 본 섹션은 이 구현 부채를 **가시화**하는 목적이다.

#### 4.9.1 Aerial Gym 공식 경로 채택 요약

본 연구는 Aerial Gym `navigation_task`의 **표준 I/O 규약**을 최대한 재사용한다:

- **로봇**: `lmf2` (registry key, `aerial_gym/robots/__init__.py`)
- **컨트롤러**: `lmf2_velocity_control` (registry key, `aerial_gym/control/__init__.py`)
- **관측**: 81-d (17-d state: goal 4 + euler 3 + lin_vel 3 + ang_vel 3 + prev_action 4) + 64-d VAE latent) — `navigation_task_config.py`의 default와 일치
- **행동**: 3-d velocity command
- **환경 백본**: Warp-based ray-caster + camera

이 채택으로 **reward shape, VAE 사전학습 체크포인트, 컨트롤러 구현, env asset 로딩**을 기존 인프라에서 재사용한다. 이는 §4.2의 "DCE와 직접 비교 가능" 주장이 실제로 성립하게 만든다 (이전 초안은 CTBR 4-d + 32-d ray-cast로 DCE와 I/O가 달랐다).

#### 4.9.2 Must-Implement Work Items (공수 추정 포함)

| 항목 | 분류 | 공수 | 설명 |
|------|------|------|------|
| **W1**: `pretrain_adapt_task_config.py` 신규 작성 | 신규 | 3~5일 | `navigation_task_config.py`를 복제 후 reward·success 조건을 통일 |
| **W2**: Custom `Policy` nn.Module (Encoder/GRU/Heads 분리) | 신규 | 5~7일 | **2개 input branch** (17-d proprio MLP + 64-d VAE projection, §4.2 Option A) + fusion + GRU + heads, Sample Factory `ActorCritic` 인터페이스 구현 |
| **W3**: Custom `LoRALinear` 모듈 | 신규 | 3~4일 | `nn.Linear` subclass, forward에서 $W + \frac{\alpha}{r}BA$ 계산. PEFT 자동 경로 대체 (§4.9.4 근거) |
| **W4**: 3개 adaptation mode를 한 클래스에서 처리하는 `build_policy(mode=...)` 팩토리 | 신규 | 2~3일 | mode ∈ {full_ft, frozen, lora}에 따라 `requires_grad` 설정 또는 LoRALinear 교체 |
| **W5**: D3 (Precision Stationkeeping) task-specific reward/termination | 신규 | 4~6일 | `docking_env`는 geometry만 제공. **"위치 0.3m + 속도 정지 + 5초 dwell"** logic 직접 추가 (이전 "yaw 제어" 설계는 velocity command action space와 호환 불가하여 제거, §5.9 참조) |
| **W6**: D4 (dynamic) 이동 장애물 구현 | 신규 | **5~7일 (재추정)** | `dynamic_environment`의 `num_env_actions=6` mechanism이 정확히 어떻게 동작하는지 코드 레벨 검증 필요. 이동 장애물 속도·패턴의 scripted policy wrapper 구현 + M1에서 이동 장애물이 학습 신호로 작동하는지 실효성 검증. 이전 2~3일 추정은 upstream 지원을 낙관했던 과소평가였음. |
| **W7**: 사전학습 환경 (`env_with_obstacles` 저난이도 curriculum) | 신규 | 2~3일 | `empty_env`는 ray-cast 미지원(`use_warp=False`)이므로 `env_with_obstacles`의 낮은 난이도를 사전학습 분포로 사용 |
| **W8**: Ablation A6 custom GRUCell + LoRA | 신규 | 5~7일 | `nn.GRU` → `nn.GRUCell` loop 교체 + 각 gate weight에 LoRALinear 적용 |
| **W9**: Sample Factory ↔ custom policy 체크포인트 I/O | 수정 | 2~3일 | SF의 state_dict 저장 형식에 맞춰 custom policy serialization |
| **W10**: Robustness eval variant configs (A7) | 신규 | 2일 | 외란·IMU 노이즈·컨트롤러 게인의 4가지 variant config 생성 |
| **W11**: Recurrent PPO gradient monitoring hook | 신규 | 1~2일 | **Custom LoRALinear + GRU 조합의 gradient 안정성 모니터링 — M1 smoke test 의무 체크리스트.** Recurrent PPO에서 hidden state 처리 실수는 gradient explosion을 유발한다 (arXiv 2205.11104, 2022). SF native recurrent가 안전하지만 custom LoRA 주입 후 재검증 필수. Per-layer gradient norm + learning_rate × grad_norm product를 매 rollout마다 기록. |

**총 추정 공수**: ~38~53일 (2~2.5 개발자-month 집중 작업). M1 (2개월)에 배정 타당하되 **M1 종료 시점 여유 없음** — 공수 압박 발생 시 A6 (W8) 포기 옵션.

#### 4.9.3 Custom LoRALinear 직접 구현 결정 (PEFT 자동 경로 포기)

**왜 PEFT `get_peft_model()`을 쓰지 않는가** — 이전 초안의 결정을 재검토했다.

Sample Factory는 `sample_factory.model.actor_critic.create_actor_critic()`으로 모델을 구성하고, checkpoint는 특정 key 이름 규칙(encoder/core/decoder 계층 이름)으로 저장한다. PEFT의 `get_peft_model()`은 모델을 `PeftModel`로 감싸면서 state_dict 키에 `base_model.model.*` prefix를 추가한다. 이는 Sample Factory의 **체크포인트 로드·저장 루틴과 직접 호환되지 않으며**, 또한 SF의 `forward()` 내부에서 수행되는 `normalize_input`, `policy_initialization`, RNN state 관리와 LoRA adapter가 적재되는 순서를 조정해야 한다.

이런 마찰을 피하기 위해 **custom LoRALinear nn.Module을 직접 작성**한다. 약 50줄 PyTorch 코드로 구현 가능하며, SF의 기존 `nn.Module` 인터페이스와 완전히 자연스럽게 통합된다.

```python
class LoRALinear(nn.Module):
    """nn.Linear drop-in replacement with low-rank adaptation."""
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r
    
    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
```

적응 모드 전환은 `build_policy(mode='lora')`에서 지정된 layer의 `nn.Linear`를 이 `LoRALinear`로 교체하는 것으로 충분하다. Verification: 각 모드 빌드 후 `sum(p.numel() for p in policy.parameters() if p.requires_grad)`를 찍어 Full-FT ~280K / Frozen ~150K / LoRA ~7K 확인.

#### 4.9.4 $\mathcal{L}_{\text{dyn}}$ 보조 손실 제거 배경

이전 초안의 dynamics prediction 보조 손실 $\mathcal{L}_{\text{dyn}}$은 **Sample Factory Learner를 수정해야 구현 가능**하다. SF의 PPO Learner는 rollout buffer에서 (obs, action, return, advantage)만 꺼내 쓰고, fusion latent $z_t$ 같은 intermediate activation은 저장하지 않는다. $\mathcal{L}_{\text{dyn}}$을 추가하려면:

- `ActorCritic.forward()`에서 $z_t$를 별도 field로 노출
- Rollout worker가 $z_t$를 buffer에 저장
- `Learner._calculate_losses()`를 override해 $z_{t+1}$ 예측 MSE 추가

이는 framework-level 수정이며 구현·디버깅 부담이 크다. 본 연구는 이득·리스크 비교 후 **보조 손실을 제거하고 순수 PPO로 사전학습**하기로 결정. 잃는 것은 사전학습된 표현의 품질 개선 가능성, 얻는 것은 구현 안정성과 $\mathcal{L}_{\text{dyn}}$ 가중치 $\lambda_d$ 튜닝 노력 절약.

#### 4.9.5 4개 다운스트림 Task의 실제 상태 및 구현 필요 사항

Reviewer 피드백에서 확인한 대로, Aerial Gym의 4개 env는 "env geometry"만 제공하고 task-level logic은 별도 구현이 필요하다.

| Task | Upstream 상태 | 추가 구현 필요 |
|------|--------------|--------------|
| **D1: Dense Forest** | `forest_env.py` (env config 존재) | task-level reward/success 공통화 (W1) |
| **D2: Corridor** | fork의 `corridor_env.py` (env config 존재) | task-level reward/success 공통화 (W1) |
| **D3: Precision Stationkeeping** | fork의 `docking_env.py` (geometry만) | **"위치 0.3m + 속도 정지 + 5초 dwell" success 조건 신규** (W5). 이전 "yaw 제어" 대안은 velocity command action space로 구현 불가하여 제거. |
| **D4: Dynamic** | `dynamic_environment.py` + `num_env_actions=6` | 이동 장애물 파라미터화 wrapper (W6, 공수 5~7일로 재추정) |

사전학습 환경은 `empty_env`를 사용하려 했으나 `empty_env`는 `use_warp=False`라 ray-cast가 동작하지 않고 장애물도 없다. 대안: **`env_with_obstacles`를 낮은 난이도(장애물 2~3개)로 설정**해 사전학습 분포로 사용한다. 이는 D1~D4와 같은 코드 경로를 공유하는 이점이 있다.

#### 4.9.6 Sample Factory 통합 세부

**Wrapper**: Aerial Gym 공식 제공 `AerialGymVecEnv` wrapper 사용 (RL Training 문서 참조).

**Custom Policy 구조**: SF의 `ActorCritic` 인터페이스를 구현하되 내부는 단일 `nn.Module`. 이는 PEFT wrapping 이슈 회피 및 custom LoRALinear와의 호환성 확보. 구체 구조는 W2 참조.

**Parallel env 설정**:
- `num_envs=4096` (Warp 백엔드, VRAM 여유 확인)
- `rollout=32` (SF `rollout` 파라미터)
- `num_batches_per_epoch=8` → 실효 minibatch $4096 \times 32 / 8 = 16,384$
- `num_epochs=5`

**GRU 관리**: SF의 `model_core_rnn` 설정은 GRU/LSTM을 native 지원하며 hidden state 관리(episode reset 시 zero-out, rollout 내 propagation)를 SF core가 담당한다. Custom policy는 GRU를 `core` 역할로 등록만 하면 됨.

#### 4.9.7 하드웨어 Budget 재추정 (Wall-clock 여유 포함)

RTX 3090 (24GB VRAM) 기준. Reviewer 지적 반영해 **wall-clock buffer를 실효 1.5~2x로 확대**.

| 항목 | 요구 / 추정 |
|------|-------------|
| Isaac Gym + Warp (4096 envs) | ~6 GB VRAM |
| Policy (~280K) + rollout buffer | ~3 GB VRAM |
| 카메라 렌더링 (270×480 depth, 4096 envs) | ~2 GB VRAM |
| **총 VRAM 사용** | **~11 GB (여유 13 GB)** |
| 이론 wall-clock 1 run | ~2.5 hours |
| **실효 wall-clock 1 run** (overhead 포함) | **~4 hours** |
| Main runs (80) × 4h | ~320 h |
| Ablations (~108) × 4h | ~430 h |
| Hyperparameter sweep (36) × 2h | ~72 h |
| Debug·재실행 buffer (30%) | ~245 h |
| **총 GPU-hours** | **~1070 h ≈ 45일 24/7** |

RTX 3090을 24/7 돌린다는 가정에서 45일. M2~M3 두 달 60일 배정이면 현실적으로 커버 가능하되 **여유가 타이트**하다. 이는 일정 재조정에 반영 (§8.1).

#### 4.9.8 Dependencies 고정

- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.8 (Isaac Gym 요구사항)
- **PyTorch**: 2.0.1 + cu118
- **Isaac Gym**: Preview 4
- **Aerial Gym**: github release tag 고정
- **Sample Factory**: 2.1.x
- **CUDA Toolkit**: 11.8

`environment.yml` / `requirements.txt` 및 `Dockerfile`을 repo에 포함해 재현 가능하게 함.

#### 4.9.9 구현 리스크 재평가 체크리스트

이전 "✓" 일변도 표를 **실제 상태를 반영한 체크리스트**로 재작성.

| 질문 | 답 | 근거 |
|------|-----|------|
| 관측 공간 81-d (17-d state + 64-d VAE)를 Aerial Gym에서 바로 구성 가능? | **✓ (기존 default)** | `navigation_task_config.py`의 default와 일치. Option A (2-branch) 적용 |
| Velocity command 3-d 행동 + `lmf2_velocity_control` 사용 가능? | **✓ (기존 예제 존재)** | DCE 예제, navigation_task 모두 이 조합 |
| 4개 downstream task가 공통 class family로 즉시 통합되는가? | **✗ 직접 구현 필요** | W1, W5, W6가 필수 (~10일) |
| Sample Factory + custom policy + recurrent PPO 동작? | **△ 참조 코드 있음** | DCE 예제 존재, custom model은 직접 작성 |
| PEFT `get_peft_model()`로 LoRA가 자동 적용? | **✗ 직접 구현 필요** | Custom LoRALinear로 대체 (§4.9.3) |
| $\mathcal{L}_{\text{dyn}}$ 보조 손실을 즉시 추가 가능? | **✗ 제거 결정** | SF Learner 수정 부담 크므로 순수 PPO로 단순화 |
| 사전학습 checkpoint → 3개 적응 전략 초기화? | **△ 직접 구현** | `build_policy(mode=...)` 팩토리 신규 (W4) |
| A6 (GRU LoRA) 실패해도 main이 완결? | **✓** | Main pipeline과 독립, negative report 가능 |
| RTX 3090 4개월 GPU budget 충분? | **△ 타이트하지만 가능** | 45일 실효, 60일 배정 |

**종합**: 이전 초안보다 **약 3주치 구현 작업이 정확히 가시화**됐다. 이는 timeline 재조정(§8.1)과 이번 방법론 단순화(VAE latent 채택, $\mathcal{L}_{\text{dyn}}$ 제거, velocity control 채택)로 수용 가능하다.

---

## 5. Experimental Setup

### 5.1 Research Questions

- **RQ1 (Paradigm Validation)**: 사전학습-적응 패러다임은 동일 총 예산의 scratch PPO 대비 성능·효율·안정성을 개선하는가?
- **RQ2 (Three-way Comparison)**: Full-FT / Input-Branch-Frozen / LoRA 세 전략은 다섯 축에서 어떻게 다른가?
- **RQ3 (Task-Dependence)**: 세 전략의 상대적 우위는 다운스트림 과업 특성(환경 복잡도, 외란 수준, 시간 구조)에 따라 어떻게 변하는가?
- **RQ4 (LoRA Internal)**: LoRA의 rank, 주입 위치, GRU 적응 여부는 성능에 어떤 영향을 주는가?

### 5.2 Simulation Environment

- **시뮬레이터**: Aerial Gym Simulator v2.0.0, **Warp 백엔드** (GPU ray-casting + depth camera rendering).
- **병렬 환경**: 4096 parallel envs (Aerial Gym의 대규모 병렬성 활용).
- **RL 라이브러리**: Sample Factory 2.1.x (Aerial Gym 공식 통합, recurrent PPO 지원, DCE 예제와 동일 stack).
- **기체**: `lmf2` (Aerial Gym registry key).
- **컨트롤러**: `lmf2_velocity_control` (Aerial Gym navigation 표준). 3-d velocity command → body rates → motor RPM 변환. 100 Hz.
- **관측**: 81-d (17-d state + 64-d VAE latent). VAE는 Aerial Gym 제공 사전학습 체크포인트 사용. 정확한 index layout은 §2.1 참조.

### 5.3 Tasks Mapped to Aerial Gym Environments

**사전학습 $\mathcal{T}^{\text{pre}}$ — Low-difficulty `env_with_obstacles` curriculum**

Aerial Gym의 `empty_env`는 `use_warp=False`로 ray-cast와 장애물 지원이 없어 본 연구의 관측 파이프라인과 호환되지 않는다. 대신 **`env_with_obstacles`의 저난이도 설정**을 사전학습 분포로 사용한다.

- Environment: `env_with_obstacles`, `num_assets=2~3` (저밀도)
- Asset types: 단순 cube 장애물
- Reward: navigation_task 표준 reward shape 재사용 (§5.9)
- 목적: 기본 비행 제어 + goal-seeking + 약한 장애물 인식

이 선택으로 사전학습과 다운스트림(D1~D4)이 **동일한 코드 경로와 observation pipeline**을 공유하게 되어 transfer 품질이 향상되고 구현 부채가 줄어든다.

**다운스트림 과업 (4개)**:

| Task | Aerial Gym Env | 신규 구현 필요 사항 | Distance from Pretrain |
|------|----------------|-------------------|----------------------|
| **D1: Dense Forest** | `forest_env` (upstream) | task-level reward/success 공통화 (W1) | In-distribution 확장 (더 높은 장애물 밀도) |
| **D2: Corridor Navigation** | `corridor_env` (fork) | task-level reward/success 공통화 (W1) | OOD 공간 구조 |
| **D3: Precision Stationkeeping** | `docking_env` geometry (fork) | **"위치 0.3m + 속도 정지 + 5초 dwell" success 조건 신규 구현 (W5)** — 이전 "yaw 15° 제어"는 action space 호환성 문제로 제거 (§5.9 참조) | OOD 과업 유형 (시간 구조: sustained hover) |
| **D4: Dynamic Obstacles** | `dynamic_environment` (upstream, `num_env_actions=6`) | 이동 장애물 파라미터화 wrapper (W6, 5개 이동 cube) | OOD 동역학 (시간 의존성 높음) |

**이 4개 과업 선택의 근거**:
- 모두 Aerial Gym의 env geometry를 재사용 가능 (task-level logic만 직접 구현)
- 사전학습 분포와의 거리 스펙트럼 형성 (D1 가까움 → D4 가장 멀음)
- 각기 다른 도전 — 정적 밀도(D1), 공간 구조(D2), 과업 유형(D3), 동역학(D4)
- RQ3(과업 특성 의존성)에 대한 풍부한 실증적 답 가능

**D3 (stationkeeping) success condition (신규 구현)**: 목표 지점 $p_g$에서 **0.3m 이내 + 속도 $\|v\| < 0.1$ m/s + 5초 dwell**. Reward는 distance 항 + goal 근방에서 활성화되는 velocity damping 항. 세부는 W5. Aerial Gym의 velocity command action space 제약 하에서 "정밀 stationkeeping"을 학습시키는 task로, D1/D2의 "goal-reach + move on" 시간 구조와 대비된다.

**D4 (dynamic) 구성 (신규 구현)**: 5개 이동 cube 장애물, 각 cube는 random waypoint 간 0.5~1.5 m/s로 이동. `num_env_actions=6` mechanism 활용해 이동 패턴 sampling. 세부는 W6.

### 5.4 Robustness Axis (Aerial Gym의 고유 강점 활용)

Aerial Gym은 다음 randomization 기능을 제공하며, 본 연구는 이를 **robustness 평가의 추가 축**으로 활용한다.

- **외란**: `disturbance.max_force_and_torque_disturbance` (6D force/torque) + `prob_apply_disturbance` (`base_quad_config.py`)
- **센서 노이즈**: IMU bias/노이즈/포화 (`imu_sensor.py`)
- **도메인 랜덤화**: 컨트롤러 게인 $K_{pos}, K_{vel}, K_{rot}, K_{angvel}$ env·reset별 randomize (`base_lee_controller.py`)

**Robustness evaluation**: 각 적응 전략으로 학습한 final policy를 다음 변형 환경에서 평가하여 robustness 비교.

- **Clean** (default): 훈련과 동일
- **Disturbance++**: 외란 확률 2배, force/torque 크기 1.5배
- **Sensor Noise++**: IMU noise 표준편차 2배
- **Controller Gain Shift**: 훈련 시와 다른 게인 범위

이 4가지 variant에서의 성능 drop 비교가 **각 적응 전략의 robustness 특성**을 드러낸다. LoRA의 저차 제약이 over-specialization을 억제해 더 robust할 것이라는 가설 검증.

### 5.5 Independent Variables

**Adaptation strategy (main, 5 seeds × 4 tasks)**:
1. **B-Scratch**: Scratch PPO
2. **B-FullFT**: Pretrained + Full Fine-Tuning
3. **B-Frozen**: Pretrained + Input-Branch-Frozen (DCE-style)
4. **Ours-LoRA-r8** (default): Pretrained + Linear-only LoRA rank 8

**Ablations (3 seeds)**:
- A1 — LoRA rank: $r \in \{1, 2, 4, 16\}$
- A2 — LoRA injection location: {fusion-only, heads-only, both (default)}
- A3 — Actor-only LoRA (vs Actor+Critic default)
- A4 — LoRA learning rate: $\{1\text{e-}3, 3\text{e-}3\}$ (Hu et al. 2021 관례의 민감도)
- A5 — 사전학습 예산 $B_{\text{pre}}^*$: $\{2.5 \times 10^7, 1 \times 10^8\}$ (pretrain 투자의 민감도)
- **A6 — LoRA + GRU**: GRU gate matrices에도 LoRA 주입 (D3, D4 한정, custom GRUCell + LoRALinear)
- **A7 — Robustness evaluation**: main 4 조건을 4가지 variant 환경에서 zero-shot 평가 (training 없음)

### 5.6 Dependent Variables (5 Axes + Robustness)

**축 1 — Performance**: Final success rate, collision rate, time-to-goal, cumulative return

**축 2 — Sample efficiency**: Steps-to-threshold (success rate 0.8 도달), learning curve AUC

**축 3 — Convergence stability**: Final performance의 seed std/IQR (5 seeds), 학습 발산 횟수

**축 4 — Hyperparameter sensitivity**: lr × entropy 3×3 sweep 결과의 IQR

**축 5 — Forgetting magnitude (2-component measure)**: 
- **(5a) Behavioral retention**: 적응 후 정책 $\pi_{\theta_{\text{adapt}}}$을 사전학습 환경 $\mathcal{T}^{\text{pre}}$에서 평가한 성능. $\text{Forgetting}_{\text{behav}} = \text{perf}_{\text{pre, before}} - \text{perf}_{\text{pre, after}}$
- **(5b) Policy KL divergence**: $D_{\text{KL}}(\pi_{\theta_0}(\cdot|s) \| \pi_{\theta_{\text{adapt}}}(\cdot|s))$를 $\mathcal{T}^{\text{pre}}$ state distribution 위에서 측정. Wołczyk et al. (2024)의 "state coverage gap" 정밀 protocol에는 못 미치지만 behavioral metric보다 세밀한 drift 신호를 제공. LoRA의 $\Delta W = \frac{\alpha}{r}BA$ 제약이 실제로 정책 분포 drift를 억제하는지 직접 검증.

**보조 축 — Robustness (A7)**: 4개 variant 환경에서의 성능 drop.

**Resource metrics**: Trainable param 수, 학습 wall-clock time, 추론 latency on Jetson Orin Nano emulation (RTX 3090 batch=1 + FP16 추정), memory footprint.

### 5.7 Controlled Variables

| 항목 | 값 |
|------|-----|
| 총 환경 상호작용 예산 $B_{\text{total}}$ | $2 \times 10^8$ env steps |
| 사전학습 예산 $B_{\text{pre}}^*$ | $5 \times 10^7$ env steps |
| 정책 아키텍처 | 동일 ~280K |
| Parallel envs | 4096 |
| Seeds | Main 5 (1, 7, 42, 123, 2024), Ablation 3 |
| Episode horizon | 1024 steps |
| 평가 | 100 episodes × 5 seeds per condition |

### 5.8 PPO Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate (Full-FT, Frozen, Scratch) | $1 \times 10^{-4}$ |
| Learning rate (LoRA) | $3 \times 10^{-4}$ |
| PPO clip $\epsilon$ | 0.2 |
| GAE $\lambda$ | 0.95 |
| Discount $\gamma$ | 0.99 |
| Entropy coef | 0.005 |
| Value coef | 0.5 |
| Max grad norm | 1.0 |
| Rollout length | 32 steps |
| Minibatch size | 16384 |
| Epochs per rollout | 5 |

### 5.9 Success Criteria and Reward

**Success** (task별 customizable):
- D1, D2: 목표 위치 0.5m 이내 진입 + 3초 dwell
- **D3 (Precision Stationkeeping, ex-docking)**: 목표 지점 **0.3m 이내 + 속도 $\|v\| < 0.1$ m/s + 5초 dwell** (W5 구현)
- D4 (dynamic): 목표 위치 0.5m 이내 진입 + 이동 장애물 회피 (5초 dwell)

**D3의 이전 "정밀 yaw 제어"에서 "velocity stationkeeping"으로의 재정의 근거**: 본 연구의 action space는 3-d velocity command이고 yaw는 `lmf2_velocity_control` 컨트롤러가 내부적으로 자동 조정한다. 따라서 yaw rate를 policy가 직접 제어할 수 없으며, "yaw 15° 정밀도"를 학습시키려면 action space를 4-d (velocity + yaw rate)로 확장해야 하는데 이는 navigation_task default와 어긋나 공정 비교가 깨진다. 대안으로 **"위치 정밀도 + 속도 정지 + dwell"** 기준을 채택하면 action space 통일을 유지하면서도 D3의 OOD 특성(**stationkeeping task type, 시간 구조 차이**)을 보존한다. 이는 UAV 실무의 "hover-at-waypoint" 시나리오와도 직접 대응.

**Failure**: 충돌 (장애물 0.2m 이내), 고도 위반 ($0.5 < z < 5$), timeout (1024 steps).

**Reward (base structure, Aerial Gym navigation_task reward shape 기반)**:
$$r_t = -\Delta\|p_t - g\|_2 + 10 \cdot \mathbb{1}[\text{success}] - 10 \cdot \mathbb{1}[\text{collision}] - 0.001 \|v_t^{\text{cmd}}\|_2^2$$

마지막 항은 velocity command의 크기 페널티로 부드러운 제어 유도. D3는 추가로 속도 정지 유도를 위해 목표 근처(0.5m 이내)에서 **$-0.1 \|v_t\|_2$** 항이 활성화됨. Reward 가중치는 사전학습과 모든 downstream task에서 공통 구조.

### 5.10 Run Count and Budget

| 실험 | Conditions × Tasks × Seeds | Runs |
|------|-----------------------------|------|
| Main | 4 × 4 × 5 | 80 |
| A1 (LoRA rank) | 4 × 4 × 3 | 48 |
| A2 (injection location) | 2 × 2 × 3 (기본 both와 비교하는 2 variants × 대표 task 2개) | 12 |
| A3 (actor-only LoRA) | 1 × 4 × 3 | 12 |
| A4 (higher lr) | 2 × 2 × 3 | 12 |
| A5 (pretrain budget) | 2 × 2 × 3 | 12 |
| A6 (LoRA+GRU) | 1 × 2 × 3 | 6 |
| A7 (robustness eval) | 4 × 4 × 4 variants (eval only, no training) | — |
| Hyperparameter sweep | 9 × 4 × 1 | 36 |
| **Total training runs** | | **~218** |

**GPU budget 재추정** (§4.9.7 반영):
- 이론 wall-clock: 평균 2.5h/run
- 실효 wall-clock (overhead 30%): 평균 3.25h/run
- Main (80) × 3.25h = 260h
- Ablations (102) × 2.5h = 255h (A2~A7 평균 더 짧음)
- Sweep (36) × 2h = 72h
- Debug/재실행 buffer 30% = 177h
- **총 ~770 GPU-hours ≈ 32일 24/7 on RTX 3090**

M2~M3 두 달(60일) 배정에서 **현실적으로 커버 가능하되 여유 타이트**. 실제 실행 시 실패 run의 재실행과 ablation 우선순위 조정이 필요할 수 있으며 §8.2에서 명시적 mitigation 계획.

### 5.11 Statistical Analysis

- 5 seeds 결과의 평균 ± 표준편차
- 조건 간 비교: Welch's t-test, $p < 0.05$
- Multiple comparison은 Bonferroni correction (4 task × 4 strategy = 16 비교 시 $p < 0.003$)
- Learning curve shaded area는 ±1 std

---

## 6. Expected Results and Interpretation

### 6.1 Figures

**Figure 1 — Framework Overview**. 사전학습 → 세 적응 전략 → 4개 다운스트림 파이프라인. 각 전략별 frozen/trainable 파라미터 색상 시각화.

**Figure 2 — Learning Curves on D1 (in-distribution)**. x축 env steps, y축 success rate. 4 main conditions × 5 seeds (shaded ±1 std). RQ1과 RQ2의 주 시각적 답.

**Figure 3 — Learning Curves on D2, D3, D4 (generalization)**. 3개 subplot. RQ3의 주 답.

**Figure 4 — Five-axis Stability Radar Plot**. 4 main conditions × 5축 (performance, efficiency, seed std, hyperparameter IQR, retention). 각 전략을 폴리곤으로 표시.

**Figure 5 — LoRA Ablation**. 3개 subplot.
- (a) Rank vs final performance: $r \in \{1, 2, 4, 8, 16\}$
- (b) Injection location: fusion-only / heads-only / both
- (c) GRU LoRA marginal effect (D3, D4에서)

**Figure 6 — Compute-Performance Pareto**. x축 trainable parameters (log), y축 final success rate 평균. LoRA rank별 점들의 frontier.

**Figure 7 — Robustness Comparison**. 4 main conditions × 4 variants (clean, disturbance++, noise++, gain shift). Heatmap 또는 grouped bar chart로 성능 drop 시각화.

### 6.2 Tables

**Table 1 — Main Quantitative Results**. 4 main conditions × 4 tasks × (success rate, collision rate, steps-to-threshold), mean±std over 5 seeds.

**Table 2 — Parameter and Compute Cost**. 조건별 trainable params, wall-clock time, inference latency, memory footprint.

**Table 3 — Hyperparameter Sensitivity IQR**.

**Table 4 — Forgetting Magnitude**. 조건별·task별 retention rate.

**Table 5 — Ablation Summary**.

**Table 6 — Robustness under Environment Variants**.

### 6.3 Hypothesized Results (pre-registered)

- **H1 (RQ1)**: 사전학습-적응 세 조건 모두 Scratch 대비 (33% 더 적은 다운스트림 예산에도 불구하고) steps-to-threshold 30~50% 단축, seed std 50%+ 감소.
- **H2 (RQ2)**: Full-FT가 D1에서 최고 성능 but 가장 높은 forgetting (behavioral + KL 양쪽); Input-Branch-Frozen은 D1, D2에서 견고 but D3에서 천장 효과; LoRA-r8이 ~2.6% trainable로 Full-FT 성능의 **70~85% (task-dependent)**를 회복, **policy KL divergence가 가장 작음** (저차 제약의 직접적 증거). 85% 상한은 D1, D2 같은 in-distribution에 가까운 task, 70% 하한은 D3, D4 같은 OOD task — 특히 GRU(~99K, 전체 35%)가 frozen이라 시간 구조 변화가 큰 task에서는 불가피한 제약.
- **H3 (RQ3)**: D1→D4로 갈수록 Input-Branch-Frozen의 단점이 노출, Full-FT/LoRA의 우위 커짐. LoRA는 항상 Pareto-optimal에 가까움.
- **H4 (RQ4)**: Rank 2~8이 sweet spot. Linear-only LoRA가 in-distribution task(D1, D2)에서는 Full-FT 85%+ 회복하나 OOD task(D3, D4)에서는 70~85% 수준으로 하락할 수 있음. A6에서 GRU LoRA는 D3, D4에 한정해 marginal 개선 (혹은 개선 없음 — 후자도 "GRU 동결로 충분"이라는 명시적 finding).
- **H5 (Robustness)**: LoRA가 가장 robust (저차 제약이 over-specialization 억제), Full-FT가 가장 fragile.

### 6.4 Defensive Interpretation for All Outcomes

본 실험 설계는 가설이 맞든 틀리든 논문이 성립하도록 구성되어 있다.

- **모든 전략이 비슷한 성능** → "UAV 항법의 pretrain-adapt은 전략 선택에 robust" finding 자체가 기여
- **LoRA 압도적 우위** → 강한 결론, 명확한 contribution
- **Frozen 우위** → DCE-style이 UAV 최적임을 정량 입증
- **Full-FT 우위** → Wołczyk 발견이 UAV에서는 약하다는 반례
- **Rank 1 충분** → SLowRL 결론의 UAV 확장

---

## 7. Contributions

### Contribution 1 — Paradigm Transplantation (Conceptual)

NLP/CV의 pretrain-then-adapt 패러다임을 UAV 항법 RL에 **명시적으로 이식**하고, 언어 학습 비유(§1.3)의 원리적 정당성에서 출발해 **총 환경 상호작용 예산 통제 하의 공정 비교**로 유효성을 정량 입증한다. 기존 UAV RL의 scratch 재학습 관행에 체계적 대안을 제시.

### Contribution 2 — First Systematic Three-way Comparison in Jetson-Scale UAV Navigation

Full-FT, Input-Branch-Frozen, LoRA를 **동일 사전학습 backbone·동일 환경·동일 예산**에서 비교한 첫 UAV RL 연구. 다섯 축(performance, sample efficiency, convergence stability, hyperparameter sensitivity, forgetting) + robustness 축 평가.

**Novelty의 정직한 위치**: NLP(Hu et al. 2021)와 quadruped(SLowRL 2026), manipulation(CORAL 2026)에서 각 전략은 개별적으로 검증되었다. 본 연구의 기여는 **"새로운 적응 알고리즘 제안"이 아니라 "이들 전략이 Jetson-scale UAV navigation이라는 특정 제약 조건에서 어떤 trade-off를 보이는가"를 체계적으로 매핑**하는 것이다. 이는 methodological novelty가 아닌 **도메인-특화 empirical 기여**이며, 본 연구의 target venue인 Drones (응용 중심 저널)에 부합한다. 더 큰 과학적 기여는 **"LLM 스케일이 아닌 ~280K 소형 정책 영역에서도 pretrain-adapt 패러다임이 의미 있는 trade-off를 드러내는가"**라는 미검증 영역의 empirical 답이다.

### Contribution 3 — Empirical Guidelines for Strategy Selection

다운스트림 과업 특성에 따른 전략 선택 가이드:
- (i) 사전학습 분포 유사도 (D1 가까움 ↔ D4 가장 멂)
- (ii) 환경 공간 구조 (open forest ↔ tight corridor ↔ precision stationkeeping)
- (iii) 시간 구조 변화 (정적 ↔ 동적 장애물)
- (iv) 외란·노이즈 수준 (Clean ↔ Disturbance++ / Noise++ / Gain Shift)

각 조합에서 "어느 전략이 Pareto-optimal인가"에 대한 경험적 지도. UAV RL 실무자의 의사결정 근거 제공.

### Contribution 4 — Open-source Aerial Gym + Sample Factory Reference Implementation

실제 구현된 코드베이스 공개:

- 새 `pretrain_adapt_task_config` + 공통 reward/success logic (W1, W5, W6 산출물)
- Custom LoRALinear 모듈 (~50줄) — PEFT가 비표준 지원하는 영역에 대한 clean 구현
- `build_policy(mode=...)` 팩토리로 3개 적응 모드 통일 관리
- 218+ runs의 raw 학습 곡선 및 분석 스크립트
- Dockerfile + `environment.yml`로 one-click 재현
- (Optional) GRU-LoRA wrapper (A6 성공 시)

이 코드는 **UAV RL + PEFT 교차 영역의 후속 연구 진입 비용**을 크게 낮춘다. 특히 "Jetson-scale 소형 recurrent RL 정책에 Linear-only LoRA가 충분한가"라는 empirical question에 대한 첫 재현 가능한 증거로서 가치.

---

## 8. Execution Plan and Risk Management

### 8.1 Timeline (4 months post thesis — revised to reflect implementation debt)

Reviewer 피드백에서 확인된 구현 공수(§4.9.2, 35~50일)와 GPU 예산 재추정(§5.10, ~32일 실효)을 반영해 일정을 재조정.

| 기간 | 작업 | 결정 게이트 |
|------|------|-------------|
| **M1 (2개월)** | **환경 구축 + 커스텀 컴포넌트 구현**: (i) Aerial Gym + Sample Factory + VAE pretrained checkpoint (ntnu-arl 공개본) 로딩 검증, (ii) W1~W11 완료 (PretrainAdaptTask, custom Policy, LoRALinear, build_policy 팩토리, D3 stationkeeping + D4 dynamic reward logic, 사전학습 env, SF checkpoint I/O, gradient monitoring hook), (iii) 1 seed로 4 main conditions × D1 smoke test. | **3개 gate 모두 통과 필수**: (a) 4 main condition이 D1에서 수렴, (b) **throughput 실측값 기준선 확정** — 4096 envs + Warp + camera 조건에서 M1 smoke test 중 실측한 env-steps/sec를 M2 main 실험의 예산 산정 기준선으로 동결 (v2 RA-L 벤치마크상 v1의 210 robots × 1,800 fps 대비 개선 보고되나 depth camera 포함 구체 수치는 smoke test 전까지 미정; 실측값이 예상보다 현저히 낮으면 4096 → 2048 envs 축소), (c) **gradient norm 안정** — 모든 전략에서 grad_norm < 10 유지 (폭주 시 GRU hidden 128→64 fallback, LoRA rank 조정). 추가로 wall-clock 실측해 M2 budget 재검증. |
| **M2 (1개월)** | Main 실험 전체 (80 runs × 평균 3.25h = ~260h). 4 tasks × 4 strategies × 5 seeds. | Main raw data 100% 확보 |
| **M3 (3주)** | Ablation A1~A7 (~102 runs ≈ 255h) + A7 robustness eval (no training) + Hyperparameter sweep (36 runs ≈ 72h) | 모든 raw data 확보 |
| **M4 (3주)** | 결과 분석, figures/tables 작성, 논문 초고, 내부 검토, Drones 제출 | 제출 |

**M3→M4 경계 fluidity 노트**: M3의 GPU 실효 소요(~325h)는 3주(~504h, 24/7 가정) 안에 이론적으로 가능하지만 **가동률 거의 100% 전제**이다. 실제로는 실패 run 재실행, log 정리, 중간 분석 등이 섞이므로 M3 ablation의 일부가 M4 앞 주로 밀려날 가능성이 있다. 이 경우 M4 분석·작성이 2주로 압축되는데, 이는 (i) figures/tables의 상당수를 M2 main 완료 시점부터 병렬 작성, (ii) 낮은 우선순위 ablation(A4, A5) 축소 옵션으로 대응한다.

**M3~M4가 압축된 이유**: M1을 2개월로 확장한 만큼 뒷부분이 타이트. 그러나 (i) 실험은 GPU가 자동 실행, (ii) figures/tables의 상당수는 M2 main 완료 시점부터 병렬 작성 가능. 실제 "결정 시간"보다 "대기 시간"이 많은 구간이므로 수행 가능.

**일정 리스크 완화 전략**:
- M1 smoke test 결과에 따라 M2 main 축소 옵션: 5 seeds → 3 seeds로 축소 시 runs 48로 줄어 GPU 시간 40% 절약
- Ablation 우선순위 명시: A1 (rank) > A6 (GRU) > A7 (robustness) > A2, A3 > A4, A5. 시간 부족 시 뒤부터 제외

### 8.2 Main Risks and Mitigations

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| M1 구현 공수 초과 (W1~W11 지연) | **중** | 높음 | W1~W11 weekly milestone 추적. 필요 시 M2로 확장 → main seed 5→3 축소 |
| Recurrent PPO 학습 불안정 | 중 | 높음 | Sample Factory 검증 hyperparameter, DCE 설정 참조. 실패 시 GRU → frame stack (k=4) fallback |
| Custom LoRALinear ↔ SF 체크포인트 I/O 마찰 | 중 | 중 | smoke test 시 명시 검증 단계. 최악 시 state_dict 수동 handling |
| LoRA + GRU (A6) 구현 실패 | 중 | **낮음** | Main 결과에 영향 없음. A6만 negative report |
| LoRA가 Full-FT 대비 크게 떨어짐 | 낮음 | 중 | Rank 확대, 주입 위치 확장 (A2). Finding 자체가 기여 |
| Reviewer "GRU LoRA 왜 안 하냐" 지적 | 중 | 중 | A6로 직접 답변, §4.7 근거 |
| Reviewer "파라미터 수 다른데 왜 공정 비교?" 지적 | **중** | 중 | §2.3 framing: inductive bias 비교. §4.8 budget 통제, §5.7 통제 변인 |
| Reviewer "Scratch가 더 많은 downstream budget을 받는다" 지적 | 중 | 낮음 | §4.8에서 명시적 방어: "pretrain은 공짜 아님"을 수치로 반영 |
| 모든 전략 비슷한 성능 | 중 | 중 | "Jetson-scale에서는 전략 선택이 robust"라는 finding도 기여 |
| GPU 시간 초과 | **중** | 중 | Seed 5→3 축소, 낮은 우선순위 ablation(A2, A3, A4, A5) 제거 |
| **VRAM 초과 (4096 envs + depth camera + VAE forward)** | **중** | **높음** | 270×480 depth × 4096 envs × 4 bytes ≈ 2.1 GB/frame. 실효 VRAM 피크 시 RTX 3090 24 GB를 초과할 수 있음. M1 throughput gate에서 실측 후 **2048 envs로 축소 옵션** 명시적으로 발동. 해상도 감축 (예: 135×240) 대안. |
| **D3 stationkeeping task 학습 미수렴** | 중 | 중 | Velocity command action space로 "velocity 정지"가 학습되지 않을 경우, success 조건을 "위치 0.5m + dwell" (vanilla goal-reach)로 완화하고 "시간 구조 OOD" 기여를 D4에 집중 |
| **D4 이동 장애물의 학습 신호 부재** | 중 | 중 | `dynamic_environment`의 `num_env_actions=6` mechanism 동작이 예상과 다르거나 학습 신호 제공 불충분 시, static obstacle density를 더 높인 task로 대체 |
| Aerial Gym environment 호환성 | 낮음 | 중 | 공식 release tag 고정, Docker image |
| VAE pretrained checkpoint 로딩 실패 | 낮음 | 높음 | ntnu-arl (Kulkarni group)이 Aerial Gym v2.0.0 릴리스에 포함해 공개한 체크포인트 사용. 접근 불가 시 저자가 직접 VAE 사전학습 (추가 1주) |
| **Custom LoRALinear + recurrent PPO의 gradient 불안정** | 중 | 높음 | W11 gradient monitoring hook 의무화 (M1 gate 3). Sample Factory native recurrent는 안전하지만 LoRA 주입 시 hidden state 처리 interference 가능. Explosion 시 LoRA rank 축소·clip value 조정 |
| Drones reviewer가 sim-only 비판 | 낮~중 | 낮음 | Future work에 sim2real 명시. NTNU 공식 Sim2Real 가이드(X500 v2 + PX4)와 경로 일치함을 본문에 제시 |

### 8.3 Drones Journal Fit

- **Scope**: navigation, autonomy, machine learning이 공식 scope에 포함 ✓
- **분량**: 15~25쪽 권고에 본 제안 18~22쪽 예상 ✓
- **Sim-only**: Drones 정기 게재 ✓
- **선행 게재 분리**: 이전 Drones 2025 논문(MGH framework, real Tello)과 주제 축 명확히 다름 ✓

### 8.4 Reproducibility Commitments

- 모든 코드, raw learning curves, 분석 스크립트 paper acceptance 시 공개
- Sample Factory + Aerial Gym 의존성 버전 고정
- Seed 명시, 음성 결과도 모두 보고

---

## Appendix A. Design Decision Notes

이 부록은 본 제안의 주요 설계 결정이 어떻게 도달되었는지 기록한다. 논문 작성 시에는 제외하고, 연구자 본인과 향후 협업자의 일관성 유지를 위한 내부 문서.

### A.1 왜 Drones를 target venue로 선택했는가

**RA-L/ICRA 검토 과정**: 초기에 RA-L을 target으로 검토했으나 다음 이유로 제외.

1. RA-L의 "방법론 novelty" 기대치가 높아, systematic comparison 논문으로 통과 확률 30~45%
2. RA-L은 실기체 검증을 선호, 본 연구는 sim-only
3. 자율비행 분야는 구조적으로 전용 venue가 부족하며, 로봇공학 venue (RA-L, T-RO, Autonomous Robots)는 aerial이 minority
4. Drones는 IF 4.8로 RA-L(IF 4.6)과 거의 동등하며, sim-only 게재가 정기적
5. 저자의 이전 Drones 2025 논문(MGH framework, real Tello)과 이번 제안(pretrain-adapt, Aerial Gym)은 주제 축이 명확히 다름 → 같은 저널 재투고 문제없음

### A.2 왜 "LoRA adapter library"나 "skill composition"이 아닌가

초기 검토에서 멀티태스크 LoRA adapter library (CORAL-style)나 skill composition 방향도 고려했으나 다음 이유로 제외.

1. CORAL (2026), MoIRA (2026), LoRASA (2025) 등 선행 연구와 overlap 크다
2. Skill library 개념은 로봇공학에서 수십 년 된 주제 (DMP, options framework)라 novelty density 낮다
3. 사용자의 원래 아이디어 (LLM-style pretrain-then-adapt + 3-way 적응 비교)를 직접 검증하는 것이 더 명확한 과학적 기여

### A.3 왜 LoRA default에서 GRU를 제외했는가

§4.7에서 본문에 논의. 핵심 요점:

1. PEFT 라이브러리 표준 지원 밖 → 커스텀 구현 리스크
2. GRU의 RL 학습 불안정성
3. 대부분 다운스트림 과업은 시간 구조 보존 → GRU 적응 불필요 가설
4. Linear-only LoRA의 strong parameter efficiency 주장
5. A6 ablation으로 GRU 적응의 marginal 효과 empirical 검증

### A.4 왜 4개 task인가

- 3개는 trade-off 패턴 관찰 어려움
- 5+개는 GPU 시간 초과 리스크
- 4개가 Aerial Gym의 잘 정의된 environments (`forest_env`, `corridor_env`, `docking_env`로 geometry 재사용하되 본 연구에서 stationkeeping task로 재정의, `dynamic_environment`)에 자연스럽게 매핑
- 사전학습 분포와의 거리 스펙트럼 형성 가능

### A.5 왜 Sample Factory인가

Aerial Gym 공식 지원 3개 프레임워크(rl-games, Sample Factory, CleanRL) 중 Sample Factory 선택 이유:

1. GRU/LSTM recurrent policy 안정 지원
2. DCE 논문이 동일 stack 사용 → 직접 비교 가능
3. VizDoom 등 POMDP 환경 검증 이력

### A.6 Critical Risks Identified via Literature Review

본 제안 작성 전 5가지 리스크를 문헌 조사로 검증:

1. **LoRA on GRU 비표준** (PEFT 공식 미지원) → Linear-only default로 회피
2. **Recurrent PPO 라이브러리 민감성** → Sample Factory 확정
3. **LoRA rank 적정 범위는 backbone 크기 의존** (verl ≥32 vs SLowRL rank 1) → rank {1,2,4,8,16} ablation
4. **RL fine-tuning의 catastrophic forgetting** (Wołczyk 2024) → forgetting을 5번째 평가 축으로 추가
5. **Pretrain-adapt의 UAV 검증 사례 부재** → 본 연구의 기여 포인트

### A.7 Aerial Gym Scope Alignment

Aerial Gym의 지원 범위를 고려해 scope 결정:

- **Within scope**: 회전익 navigation (velocity command), 외란/노이즈 robustness, 동적 장애물, 대규모 병렬 RL, depth camera + VAE latent 관측
- **Out of scope**: 고정익/VTOL (Aerial Gym 미지원), 다기체/편대 (미지원), 공간적 wind field (미지원, 균일 외란으로 근사)

본 연구의 scope는 Aerial Gym이 가장 강점을 보이는 영역 — `navigation_task`의 표준 경로 — 에 정확히 부합한다. 이전 초안(custom 32-d ray-cast + CTBR 4-d)은 Aerial Gym 표준에서 벗어나 구현 부채가 컸으나, 현재 설계(81-d VAE latent + velocity command 3-d)는 기존 인프라를 최대 재사용한다.

### A.8 왜 flat 정책인가 (계층적 추상화 vs. 본 연구의 선택)

본 연구는 **flat 정책**을 사용한다 — 관측에서 velocity command까지 하나의 신경망이 직접 매핑. 대안으로 **계층적 추상화**(상위 신경망이 waypoint/trajectory를 생성, 하위 신경망이 velocity·attitude·rates로 변환)가 있다. 후자는 본질적으로 **월드모델의 잠재공간** 아이디어와 유사하다 — RL이 다뤄야 할 행동 공간 추상화 수준을 높이면 학습 난이도가 낮아진다.

계층적 추상화는 매력적이지만 본 연구에는 직접 도입하지 않았다.

1. **Scope 확산 위험**: 계층 분해를 도입하면 "어떤 수준에서 분해할 것인가", "상위/하위 경계를 어떻게 설정할 것인가" 같은 별개의 설계 질문이 논문 중심이 되어, 본 연구의 핵심 질문(3-way 적응 전략 비교)이 흐려진다.
2. **공정 비교 어려움**: 계층 구조를 도입하면 Full-FT/Frozen/LoRA가 "어느 수준에서 적용되는지"가 조건별로 달라져 비교의 명확성이 떨어진다.
3. **보완 관계**: 계층적 추상화와 본 연구의 pretrain-adapt 패러다임은 **상호 배타적이지 않다**. 계층 구조 위에서도 pretrain-adapt가 적용 가능하며, 오히려 후속 연구의 자연스러운 확장 방향이다.

따라서 계층적 추상화는 **본 연구의 scope에서 제외하되, future work로 명시**한다 (Appendix B 참조). 참고로 본 연구의 현재 설계도 이미 **얕은 계층 분해**를 포함한다 — VAE가 270×480 depth image를 64-d latent로 압축하는 것이 한 층의 추상화이며, velocity command를 `lmf2_velocity_control`이 rates/motor RPM으로 변환하는 것이 또 한 층이다. 본 연구의 novelty 축은 이 "perception + control" 인프라 사이의 **policy 네트워크 적응 전략**에 집중된다.

### A.9 Assumptions That May Need Revision Later

- $B_{\text{total}} = 2 \times 10^8$은 DCE의 실험 규모에서 추정. 실제 smoke test 후 조정 필요할 수 있음.
- $B_{\text{pre}}^* = 5 \times 10^7$는 사전학습 plateau 가정. 실측 결과 plateau가 더 빠르거나 늦을 수 있음 (A5 ablation에서 $2.5 \times 10^7$ 과 $1 \times 10^8$ 민감도 검증).
- LoRA lr $3 \times 10^{-4}$는 Hu et al. 2021 관례. RL 맥락에서 재튜닝 여지 (A4 ablation).
- GRU hidden=128은 DCE의 64보다 큼. 학습 불안정성 발생 시 M1에서 64로 축소 고려. NTNU Semantic-driven DRL (2025)이 hidden=512로 Aerial Gym에서 실기체 검증한 선례가 있어 128은 범위 내.
- Budget 비대칭(Scratch $2 \times 10^8$ vs Pre+adapt $1.5 \times 10^8$)이 reviewer 해석에 미치는 영향은 실험 결과에 따라 추가 분석 필요할 수 있음.
- **정책 크기 ~280K는 "Jetson-scale 상한"** — Depth Transfer (arXiv 2505.12428)의 VAE+LSTM+RL 구조와 DCE (ICRA 2024, ~213K)의 중간 지점. RAPTOR (2025, arXiv 2509.11481)가 2,084 params로 foundation policy 가능함을 보였으므로 **더 작은 영역(~50K~100K)에서 LoRA trade-off 검증은 future work**이다. 본 연구는 "~280K 소형 영역의 첫 체계적 매핑"으로 포지션.

### A.10 Reviewer Feedback Response Trace

본 제안의 초기 버전은 중간 검토에서 **Aerial Gym v2 코드베이스에 대한 사실 확인의 부정확함**과 **구현 공수 과소평가**를 지적받았다. 이 섹션은 해당 피드백에 대한 응답 변경 사항을 추적한다 (연구자 본인의 일관성 유지 및 향후 설계 결정의 근거 보존 목적).

**주요 수정 사항**:

| 피드백 항목 | 수정 내용 | 반영 섹션 |
|-------------|-----------|-----------|
| 45-d raw ray-cast 관측은 Aerial Gym의 기존 경로와 불일치 | 81-d (13+4+64 VAE latent)로 변경. `navigation_task_config.py` default와 일치 | §2.1, §4.2, §5.2 |
| `base_quad` + `LeeRatesController` registry key가 실제로는 navigation_task default인 `lmf2` + `lmf2_velocity_control`을 사용. VAE depth latent가 동작하려면 `lmf2` 로봇(카메라 내장)이 필수 | 실제 registry key로 전면 교체. CTBR 4-d → velocity command 3-d 전환. `base_quadrotor_with_lidar_lowres`는 카메라 없어 VAE 경로 빌드 불가, 수정 라운드에서 재교정. | §2.1, §4.2, §5.2 |
| 4개 downstream task가 "env config만 존재", task-level reward/success는 직접 구현 필요 | W1, W5, W6를 명시적 신규 구현 항목으로 분리. D3 docking success 조건을 명시적으로 정의 | §4.9.2, §4.9.5, §5.3, §5.9 |
| PEFT `get_peft_model()`과 Sample Factory의 state_dict 호환성 마찰 | PEFT 자동 경로 포기, Custom LoRALinear 직접 구현. 50줄 코드 sketch 포함 | §4.9.3 |
| $\mathcal{L}_{\text{dyn}}$ 보조 손실은 Sample Factory Learner 수정 필요 | 보조 손실 제거, 순수 PPO로 사전학습 단순화 | §4.3, §4.9.4 |
| `empty_env`는 `use_warp=False`라 본 연구의 ray-cast observation과 호환 안 됨 | 사전학습 환경을 `env_with_obstacles` 저난이도 curriculum으로 변경 | §5.3 |
| 파라미터 magnitude 차이 (280K vs 7K, 40배)가 "공정 비교" framing과 긴장 | Framing 재정의: "동일 파라미터 수 비교"가 아닌 "동일 backbone 위의 서로 다른 inductive bias 비교". §2.3 명시 | §2.3, §4.7 |
| Budget 비대칭 (Scratch 2e8 vs Pre+adapt 1.5e8)이 reviewer 비판 대상이 될 수 있음 | §4.8에서 명시적 방어: "pretrain은 공짜 아님"이라는 의도적 설계 | §4.8, §8.2 |
| GPU budget wall-clock 낙관적 추정 (24/7 24일) | Overhead 30%, debug 30% buffer 포함해 재계산 (~32일) + 일정 재조정 | §4.9.7, §5.10, §8.1 |
| M1 공수 (1개월)가 구현 부채(35~50일)를 반영 못함 | M1을 2개월로 확장, M3~M4 압축, ablation 우선순위 명시 | §8.1 |
| Contribution 2의 "first fair comparison" framing이 methodological novelty 부족 | "도메인-특화 empirical 기여"로 정직하게 재위치, "Jetson-scale 영역 미검증"이라는 더 정밀한 기여로 전환 | §7 Contribution 2 |
| Forgetting 측정이 Wołczyk 2024 protocol보다 약함 | Behavioral retention + policy KL divergence의 2-component 측정으로 보강 | §5.6 축 5 |
| Off-policy 배제가 "PPO 중심 결론이 artifact"라는 의심 여지 | 초기에 A8 sanity run (SAC on D1, 1 seed)을 추가했다가, `our_algorithms/` 배제 결정 후 SF 내 SAC 구현 공수가 1~2주로 재평가되어 제거. 대신 §2.5에 "PPO only" scope를 명시적으로 못 박고 §6.4 defensive framing으로 방어 | §2.5, §5.5 |
| Jetson-scale 배포 제약이라는 core motivation이 누락 | §2.2 "Jetson-Scale Onboard Budget"으로 승격, FP32/FP16 footprint 명시, §1.5 실무 가치, §7 Contribution 2에 반영 | §1.5, §2.2, §7 |
| 로봇 `base_quadrotor_with_lidar_lowres`는 카메라 없음 (enable_camera=False 상속) → VAE latent 채움 불가, 빌드 자체 실패 | navigation_task default인 `lmf2` 로봇 + `lmf2_velocity_control` 컨트롤러로 전역 교체. 이 조합이 VAE 경로와 본질적으로 일치 | §2.1, §4.9.1, §5.2 |
| §4.2의 3-branch 분해(proprio 13 + goal 4 + VAE 64)가 navigation_task.py의 실제 obs 인덱스 layout과 불일치 | Option A 채택: 17-d 전체를 단일 proprio MLP로 통합 + VAE 64-d와 concat하는 2-branch 구조로 단순화. Fusion 입력 160 → 128로 축소, LoRA trainable params 재계산(7,168) | §2.1, §4.2, §4.6 |
| "Encoder-Frozen" 명명에서 "encoder"가 VAE인지 policy 내부 input MLP인지 모호 | "Input-Branch-Frozen"으로 rename. §2.3 표에 "외부 VAE 공통 frozen + policy 내부 input-branch MLP 추가 frozen" 명시 | §2.3, §4.5, 제목 |
| A8 SAC sanity 공수 추정(2~3일)이 our_algorithms/ 배제 전제 하에서는 비현실적(SF 내 SAC 구현 1~2주) | A8 완전 삭제. §2.5에 "PPO only" scope 명시. §6.4 defensive framing이 이미 algorithm-artifact 우려를 커버 | §2.5, §5.5, §5.10 |
| VAE 체크포인트 출처를 "ArUco lab"으로 잘못 기재 | 실제 출처인 ntnu-arl (Kulkarni group, Aerial Gym v2.0.0 릴리스 포함)로 정정 | §2.1, §8.2 |
| Sample Factory 공식 예제 default `rnn_size=64` 대비 제안서는 128 사용 → 근거 필요 | §4.2에 "DCE의 64 대비 2배 확장 근거" 명시 (OOD task의 belief tracking 여유 + M1 smoke test 결정 게이트로 64 fallback 보류) | §4.2, §A.9 |
| H4의 "LoRA가 Full-FT 85%+ 회복" 하한이 optimistic (GRU 99K/35% frozen 고려 시 D3, D4에서 하락 불가피) | H4 하한을 **70~85% task-dependent**로 조정. H2에도 반영 | §6.3 H2, H4 |
| M3 3주 배정은 가동률 100% 전제 → M3→M4 경계가 타이트 | §8.1에 "M3→M4 fluidity 노트" 명시. 병렬 작성 전략 및 축소 옵션 구체화 | §8.1 |
| (3차 검토 - 문헌 기반 비판적 검토) Jetson Orin Nano 배포 가능성의 구체적 플랫폼 정합성 부재 | §2.2에 Holybro X500 v2 + Pixhawk Jetson Baseboard + Aerial Gym 공식 Sim2Real 가이드(X500 v2 기준) 연결. "Jetson-scale motivation은 상업적으로 확립된 플랫폼"임을 명시 | §2.2, §8.2 |
| (3차) GRU hidden=128의 근거가 "DCE 대비 확장"만 있고 최신 참조점 부재 | NTNU Semantic-driven DRL (2025)의 hidden=512 실기체 검증 사례 추가. 128을 "보수적 중간값"으로 재위치 | §4.2, §A.9 |
| (3차) Recurrent PPO + custom LoRALinear 조합의 gradient 안정성 리스크 (arXiv 2205.11104의 gradient explosion 경고) 미반영 | W11 gradient monitoring hook 신규 work item 추가. M1 smoke test의 세 번째 gate로 "grad_norm < 10" 조건 명시. §8.2에 "Custom LoRALinear + recurrent gradient 불안정" 리스크 독립 항목 추가 | §4.9.2 W11, §8.1, §8.2 |
| (3차) D3 "yaw 15° 제어" success 조건이 velocity command 3-d action space와 호환 불가 (yaw는 `lmf2_velocity_control` 내부 자동 조정) | D3를 **"Precision Stationkeeping"** (위치 0.3m + 속도 정지 + 5초 dwell)으로 재정의. Action space 통일 유지하면서 OOD 특성(sustained hover) 보존 | §5.3, §5.9, §4.9.2 W5, §4.9.5, §A.4, §7 |
| (3차) W6 (D4 dynamic obstacles) 공수 2~3일 추정이 upstream 지원 낙관 | 공수 5~7일로 재추정. `num_env_actions=6` mechanism 코드 레벨 검증 필요성 명시. M1 우선 검증 항목 | §4.9.2 W6 |
| (3차) M1 smoke test에 "Aerial Gym v2 실측 throughput" gate 부재 | "≥100K env-steps/sec with 4096 envs + Warp + camera" gate 추가. 미달 시 2048 envs 축소 옵션 발동 | §8.1 M1 |
| (3차) 4096 envs × 270×480 depth × 4 bytes ≈ 2.1 GB frame buffer로 인한 VRAM 초과 가능성이 독립 리스크로 인식되지 않음 | §8.2에 독립 리스크 항목 추가. 해상도 감축·환경 수 축소 대안 명시 | §8.2 |
| (3차) D3 task 학습 미수렴, D4 이동 장애물 학습 신호 부재 등 task-level 리스크 부재 | §8.2에 각각 독립 리스크 항목 추가. 각 완화 전략 명시 | §8.2 |
| (3차) "280K는 소형" 주장이 RAPTOR 2K 대비 135배 크다는 비판 대응 부재 | §A.9에 "280K는 Jetson-scale 상한, ~50-100K의 더 작은 영역은 future work" 명시. 본 연구를 "~280K 소형 영역의 첫 체계적 매핑"으로 포지션 | §A.9 |
| (4차 내부 일관성) §4.9.1·§4.9.9·§5.2의 obs 기술이 여전히 "13+4+64" 3-branch 표기로 남아 있어 §2.1/§4.2에서 확정한 Option A (17-d state + 64-d VAE, 2-branch)와 불일치 | 세 곳 모두 "17-d state + 64-d VAE" 표기로 통일 | §4.9.1, §4.9.9, §5.2 |
| (4차) §4.9.2 W2 설명이 "3개 input branch"로 남아 Option A와 모순 | "2개 input branch (17-d proprio + 64-d VAE projection)"로 수정 | §4.9.2 W2 |
| (4차) §8.2 리스크 표에 W11 추가 후 "W1~W9 지연" 표기가 인덱스 미갱신 | "W1~W11 지연"으로 수정 | §8.2 |
| (4차) §4.7 GRU 제외 근거 2 "본 연구의 다운스트림 과업은 주로 동일한 시간 구조"가 §5.3의 D3·D4 시간 구조 OOD 기술과 자기모순 | 근거 2를 "D1·D2만 동일 시간 구조 공유, D3·D4는 A6 ablation에서 empirical 검증"으로 톤 조정. A6를 contribution 3 (task 특성별 전략 선택)과 직접 연결 | §4.7 |
| (4차) M1 gate (b)의 "≥100K env-steps/sec" 수치가 Aerial Gym v2 벤치마크 근거 없이 임의로 선정 | "실측값을 M2 예산 기준선으로 확정" 톤으로 완화. v2 RA-L은 v1 대비 개선만 보고하고 구체 수치 미공개임을 명시 | §8.1 M1 |

**핵심 교훈**: 초기 제안은 과학 aim은 방어 가능했으나 **구현 명세가 Aerial Gym 실제 코드와 불일치**했다. 이번 개정에서 모든 구현 항목을 "직접 구현해야 할 작업"으로 재분류했으며, 이에 따라 구현 부채가 정확히 가시화되었다(§4.9.2 work items). 이 수정은 제안의 과학적 방어력을 유지하면서 **실행 가능성의 신뢰도**를 크게 높였다.

---

## Appendix B. Known Limitations

- **Sim-only**: 실제 UAV에서 동일한 trade-off 관찰 미보장, LoRA의 UAV sim2real은 미검증
- **Jetson-scale backbone에 한정**: 본 연구는 ~280K recurrent + PPO 조합, FP32 ~1.1 MB 크기에 국한. 이 크기는 배포 타당성 측면에서 의도된 제약이지만, 큰 backbone(>1M)에서는 LoRA의 상대적 우위가 더 커질 가능성. 대형 backbone 확장은 future work.
- **Flat 정책에 한정**: 본 연구는 관측 → velocity command를 직접 매핑하는 flat 정책만 비교한다. 상위 신경망이 waypoint·trajectory 같은 고수준 명령을 생성하고 하위 신경망이 이를 실제 제어 명령으로 변환하는 **계층적 추상화** (월드모델의 잠재공간 개념과 유사)는 본 연구의 3-way 적응 비교와 **상호 보완적** 방향이며, 본 연구의 자연스러운 확장으로 future work에서 다룰 가치가 있다.
- **VAE latent를 frozen 외부 모듈로 전제**: 관측 단계의 사전학습(VAE)과 policy 단계의 사전학습을 분리해 비교를 깔끔하게 했으나, 이는 "VAE도 적응시킨다면 어떻게 되는가"라는 질문을 닫아둔 것. End-to-end 전체를 적응 대상으로 확장하는 것이 future work.
- **LoRA의 RL-specific 이론 부재**: LoRA는 supervised loss 가정, RL의 비정상 reward·exploration과의 상호작용 이론 분석은 scope 외
- **Forgetting 측정의 한계**: 본 연구는 behavioral retention + policy KL divergence로 보강했으나, Wołczyk et al. 2024의 state coverage gap 정밀 protocol(state distribution divergence 측정)보다는 약함
- **4-task 일반화**: 4개 task로 trade-off 일반화는 위험할 수 있음. 5+ task 확장은 future work
- **Aerial Gym 자체의 scope 제한**: 고정익, VTOL, 다기체 실험은 이 시뮬레이터에서 본질적으로 불가. 다른 시뮬레이터(AirSim 등) 사용 시 별도 연구

---

## Appendix C. Main References

1. Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.
2. Kulkarni, M., & Alexis, K. (2024). Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding. *ICRA*. arXiv:2402.03947.
3. Kulkarni, M., Rehberg, W., & Alexis, K. (2025). Aerial Gym Simulator. *IEEE RA-L 10(4):4093-4100*.
4. Wołczyk, M., et al. (2024). Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem. *ICML*. arXiv:2402.02868.
5. SLowRL authors (2026). Safe Low-Rank Adaptation Reinforcement Learning for Locomotion. arXiv:2603.17092.
6. Zhou, Y., et al. (2024). Multi-Task Reinforcement Learning for Quadrotors. arXiv:2412.12442.
7. Fan, X., et al. (2025). Flying in Highly Dynamic Environments with End-to-end Learning. arXiv:2503.14352.
8. Petrenko, A., et al. (2020). Sample Factory: Egocentric 3D Control at 100000 FPS. *ICML*. arXiv:2006.11751.
9. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
10. Henderson, P., et al. (2018). Deep Reinforcement Learning that Matters. *AAAI*.
11. Dohare, S., et al. (2024). Loss of plasticity in deep continual learning. *Nature 632*.
12. CORAL authors (2026). Scalable Multi-Task Robot Learning via LoRA Experts. arXiv:2603.09298.
13. (Author prior work, to be self-cited): Lee, U., Lee, S., & Kim, K. (2025). Data-Efficient Reinforcement Learning Framework for Autonomous Flight Based on Real-World Flight Data. *Drones 9(4), 264*.

---

*End of proposal.*

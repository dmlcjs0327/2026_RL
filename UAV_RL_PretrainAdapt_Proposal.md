# Pretrain-then-Adapt for UAV Reinforcement Learning

**A Systematic Comparison of Full Fine-Tuning, Encoder-Frozen, and LoRA Adaptation Strategies in Aerial Gym**

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

본 연구는 자율 무인 항공기(UAV)의 학습 기반 항법을 위한 강화학습(RL) 정책 훈련에서, **기반 정책을 한 번 사전학습한 후 다운스트림 비행 과업에는 효율적 적응 전략으로 소량 조정**하는 패러다임의 유효성과 세 가지 적응 전략(Full Fine-Tuning, Encoder-Frozen, LoRA) 간 trade-off를 Aerial Gym 시뮬레이션에서 체계적으로 분석한다.

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

- **Kulkarni & Alexis (2024, ICRA, "Deep Collision Encoding")**: Aerial Gym에서 depth encoder 사전학습 후 동결, 위에 RL 정책. **Encoder-Frozen 단일 전략**.
- **SLowRL (2026, arXiv 2603.17092)**: Unitree Go2에서 sim2real LoRA. **Quadruped, 단일 task**.
- **CORAL (2026, arXiv 2603.09298)**: VLA 모델에서 per-task LoRA. **Manipulation 도메인**.

**누구도 하지 않은 것**: UAV 항법 도메인에서 Full Fine-Tuning, Encoder-Frozen, LoRA 세 전략을 **총 환경 상호작용 예산을 일치시킨 공정 조건에서 직접 비교**하고, 각 전략의 trade-off를 다양한 다운스트림 과업 특성에 걸쳐 체계적으로 매핑한 연구.

### 1.5 What This Work Delivers to Practitioners

- 새 임무 적응 시 full retraining을 회피할 수 있는 검증된 경로
- 자기 과업 특성에 맞는 적응 전략 선택의 과학적 근거
- 온보드 배포를 염두에 둔 소형 정책 크기 내에서 효율적 적응 가능성 입증
- Aerial Gym 기반 재현 가능한 오픈소스 protocol

---

## 2. Problem Formulation

### 2.1 Setting

Aerial Gym 시뮬레이터에서 관측 공간과 행동 공간을 공유하는 UAV 항법 과업군 $\{\mathcal{T}_i\}$를 상정한다.

- **관측 공간** $\mathcal{O}$ (45차원): 자세 quaternion (4) + 선속도 (3) + 각속도 (3) + 목표 상대 위치 (3) + ray-cast depth (32)
- **행동 공간** $\mathcal{A}$ (4차원 continuous): Collective Thrust + Body Rates (CTBR), 50 Hz
- **동역학**: Aerial Gym 표준 `base_quad` (under-actuated planar quadrotor)

### 2.2 Base Policy

소형 recurrent policy $\pi_{\theta_0}: \mathcal{O} \times \mathcal{H} \rightarrow \mathcal{A}$ — 약 280K 파라미터. 사전학습 과업 $\mathcal{T}^{\text{pre}}$에서 PPO + 보조 손실로 학습. $\mathcal{H}$는 GRU hidden state.

### 2.3 Three Adaptation Strategies

다운스트림 과업 $\mathcal{T}_i^{\text{down}}$에 대해 비교하는 세 전략:

| 전략 | Frozen | Trainable | Trainable Ratio | Inductive Bias |
|------|--------|-----------|-----------------|----------------|
| **Full-FT** | — | 전체 | 100% (~280K) | 최대 유연성, prior overwrite 위험 |
| **Encoder-Frozen** | proprio + extero encoder | fusion + GRU + heads | ~56% (~157K) | 지각 표현 보존, 정책만 특화 |
| **LoRA** | 전체 backbone | fusion·heads Linear의 low-rank update | ~2.5% (~7K @ r=8) | 저차 제약을 통한 prior 보존 |

LoRA의 default 주입 범위는 **Linear layer에 한정** (fusion MLP 2개 + actor/critic head 첫 Linear). GRU는 동결. (이 결정의 근거와 공정성 방어는 §4.7과 부록 A에서 논의.)

### 2.4 Core Research Question

> **소형 UAV 항법 정책에서, 총 환경 상호작용 예산 $B_{\text{total}} = B_{\text{pre}} + B_{\text{down}}$을 모든 조건에서 동일하게 통제한 공정 비교 하에, 세 적응 전략은 다음 다섯 축에서 각각 어떻게 다른가:**
>
> **(i) 샘플 효율, (ii) 최종 성능, (iii) 수렴 안정성, (iv) 하이퍼파라미터 민감도, (v) 사전학습 능력 보존도. 그리고 이 차이는 다운스트림 과업의 어떤 특성에 따라 변하는가?**

### 2.5 Scope

- **Aerial Gym 시뮬레이션 전용.** 실기체 sim2real은 명시적 future work.
- **회전익 단일 기체.** Aerial Gym 지원 하의 `base_quad`. 고정익/VTOL은 Aerial Gym 자체가 미지원이므로 scope 외.
- **State-based 관측 + 32차원 ray-cast.** Full RGB depth image는 별도 연구 영역.
- **PPO 단일 알고리즘.** SAC/TD3 off-policy와의 비교는 scope 외.
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

- **Kulkarni & Alexis (2024, ICRA, DCE)**: Depth encoder supervised pretraining 후 **frozen**. Aerial Gym 공식 navigation 예제. **Encoder-Frozen의 전형**.
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

> **UAV RL에서 사전학습 후 세 가지 주요 적응 전략(Full-FT / Encoder-Frozen / LoRA)을, 동일 기반 정책·동일 다운스트림 환경·동일 총 예산 하에서 공정 비교하고, 각 전략의 trade-off를 다양한 다운스트림 과업 특성(환경 복잡도, 시간 구조, 외란 수준)에 걸쳐 체계적으로 매핑한 연구가 없다.**

본 연구가 이 공백을 겨냥한다.

---

## 4. Proposed Method

### 4.1 Overall Pipeline

**Phase 1 — Pretraining.** 단순한 goal-reaching 환경에서 기반 정책 $\pi_{\theta_0}$를 PPO + 보조 손실로 학습.

**Phase 2 — Downstream Adaptation.** 각 다운스트림 과업에 대해 $\theta_0$를 출발점으로 세 적응 전략을 PPO로 적응시킨다. **세 전략 모두 동일한 PPO hyperparameter, 동일한 환경 상호작용 예산, 동일한 평가 protocol 사용.**

### 4.2 Policy Architecture (~280K params)

DCE (Kulkarni & Alexis 2024)와 비교 가능한 규모로 의도적 설계.

```
Observation (45-d)
├── Proprioceptive (13-d): quat(4) + lin_vel(3) + ang_vel(3) + goal_rel(3)
│       └── MLP: 13 → 64 → 64                                       [~5K params]
├── Exteroceptive (32-d): ray-cast depth
│       └── MLP: 32 → 64 → 64                                       [~6K params]
│
└── Concatenate (64+64 = 128)
        └── Fusion MLP: 128 → 128 → 128                             [~33K params]
            └── GRU: hidden=128, 1 layer                            [~99K params]
                ├── Actor head: 128 → 64 → 8 (mean+logstd × 4)      [~9K params]
                └── Critic head: 128 → 64 → 1                       [~9K params]

Total trainable: ~280K, activation: ELU
```

**설계 근거**:
- DCE와 비교 가능한 규모 → 직접 비교
- GRU 포함 → POMDP 환경 (부분 ray-cast 관측)의 belief tracking
- ~280K → Jetson Orin Nano 추론 가능한 소형 크기

### 4.3 Pretraining Objective

$$\mathcal{L}_{\text{pre}} = \mathcal{L}_{\text{PPO}} + \lambda_d \mathcal{L}_{\text{dyn}}$$

- $\mathcal{L}_{\text{PPO}}$: 표준 PPO clipped objective + value loss + entropy bonus
- $\mathcal{L}_{\text{dyn}}$: fusion latent $z_t$ 위에 작은 MLP $g_\phi$를 두어 $\hat{z}_{t+1} = g_\phi(z_t, a_t)$ MSE 예측. 표현에 dynamics-relevant 정보 유도.
- $\lambda_d = 0.1$

### 4.4 Strategy 1 — Full Fine-Tuning

$\theta_0$를 초기값으로 모든 파라미터 unfrozen, 표준 PPO로 다운스트림 보상 최대화. Trainable: ~280K.

### 4.5 Strategy 2 — Encoder-Frozen

proprioceptive encoder + exteroceptive encoder 동결 (`requires_grad=False`), fusion + GRU + heads는 학습. Trainable: ~157K. DCE 스타일.

### 4.6 Strategy 3 — LoRA (Linear-only default)

전체 backbone 동결 후 다음 위치에 rank-$r$ LoRA 주입:

- Fusion MLP의 2개 Linear (128×128, 128×128)
- Actor head 첫 Linear (128×64)
- Critic head 첫 Linear (128×64)
- **GRU: frozen (LoRA 없음)** — 근거는 §4.7

**LoRA 수식**: 각 frozen weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$에 대해
$$W' = W + \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, A \in \mathbb{R}^{r \times d_{\text{in}}}$$

- $A$: Kaiming uniform 초기화, $B$: zero 초기화 → 적응 시작 시 $\Delta W = 0$
- $\alpha/r = 2$, default $r = 8$
- **Actor와 Critic 모두 LoRA** (SLowRL 권고: actor-only 시 value mismatch)

**Trainable params (r=8)**: Fusion 2 × (128×8 + 8×128) + Actor/Critic heads = **~7K (~2.5%)**.

### 4.7 GRU Adaptation as Ablation

**GRU를 default LoRA 대상에서 제외하는 근거**:

1. **구현 표준성**: PyTorch `nn.GRU`는 cudnn packed weights를 사용해 PEFT 라이브러리의 표준 지원 대상이 아니다. Linear-only는 표준 API로 안정 구현.
2. **이론적 합리성**: 본 연구의 다운스트림 과업은 주로 **동일한 시간 구조** 하에서 환경 복잡도·외란만 변한다. GRU의 belief tracking 기능은 보존 가치가 있다.
3. **RL에서 GRU의 학습 불안정성**: GRU는 BPTT 불안정성, gradient vanishing/exploding으로 RL에서 다루기 까다로운 모듈. LoRA 저차 제약이 gate dynamics를 왜곡할 가능성.
4. **Parameter efficiency 주장 강화**: Linear-only LoRA가 ~2.5% trainable로 Full-FT의 90%+ 성능을 달성하면 강한 결론.

**GRU 적응의 효과는 Ablation A6에서 직접 검증** — 시간 구조 변화가 큰 task(D3, D4)에 한정해 LoRA+GRU 조건을 측정해 marginal contribution 정량화.

**공정 비교 이슈**: Full-FT와 Frozen은 GRU를 학습시키지만 LoRA default는 동결한다. 이는 논문의 framing이 "어느 파라미터 영역을 어떻게 적응시키는 것이 효율적인가"라는 것이지 "모든 조건이 정확히 동일한 파라미터를 학습"하는 것이 아니므로 타당하다. A6 ablation이 이 질문을 empirical로 방어한다.

### 4.8 Fair Comparison Protocol

**총 환경 상호작용 예산** $B_{\text{total}} = B_{\text{pre}} + B_{\text{down}}$ **을 모든 조건에서 동일 통제**. 이는 본 논문이 RL 적응 비교 문헌에서 자주 누락되는 critical confound를 정면으로 다루는 부분이다.

| 조건 | $B_{\text{pre}}$ | $B_{\text{down}}$ |
|------|------------------|-------------------|
| Scratch | 0 | $B_{\text{total}}$ |
| Pre + (Full-FT / Frozen / LoRA) | $B_{\text{pre}}^*$ | $B_{\text{total}} - B_{\text{pre}}^*$ |

구체 수치: $B_{\text{total}} = 2 \times 10^8$ env steps, $B_{\text{pre}}^* = 5 \times 10^7$ env steps (사전학습 plateau), 다운스트림 $1.5 \times 10^8$ env steps.

**왜 이 통제가 중요한가**: 사전학습이 "공짜 데이터"가 아니라 명시적 비용이라는 점을 인정해야 fair comparison이 성립한다.

### 4.9 Implementation Specification (Aerial Gym v2 Code-Level Mapping)

제안 방법을 Aerial Gym v2의 실제 코드베이스에서 어떻게 구현할지 명시한다. 이는 smoke test 시의 구조적 충돌을 방지하고, 재현성을 확보하며, 4개월 일정 내 완료 가능성을 보증하기 위함이다.

#### 4.9.1 Aerial Gym Task Pattern과의 통합

Aerial Gym v2는 **Gymnasium-compatible Task 패턴**을 따른다. Task는 `task_registry.make_task(task_name=...)`로 생성되며, `task_config`를 통해 `observation_space_dim`, `action_space_dim`을 노출한다. 모든 컴포넌트가 **Global Tensor Dictionary (GTD)**라는 공유 메모리 뱅크에서 in-place tensor 연산을 수행한다 (Kulkarni et al. 2025, RA-L).

**본 연구의 통합 방식**:

```
aerial_gym_simulator/
└── aerial_gym/
    └── config/
        └── task/
            ├── navigation_task_config.py         # (기존, 참조)
            └── pretrain_adapt_task_config.py     # (신규, 저자 추가)
```

신규 `pretrain_adapt_task_config.py`는 다음을 정의한다.

```python
class PretrainAdaptTaskConfig:
    # Observation: 45-d (13 proprio + 32 ray-cast depth)
    observation_space_dim = 45
    action_space_dim = 4                 # CTBR: thrust + body rates
    num_envs = 4096                       # Warp backend
    
    class robot_config:
        name = "base_quad"
        controller = "LeeRatesController"  # CTBR interface
        
    class sensor_config:
        ray_cast = {"num_rays": 32, "horizontal_fov": 360, 
                    "angular_resolution": 11.25}  # 1개 수평 링
        imu = {"enabled": True}
    
    class env_config:
        # Task별로 override:
        # pretrain: empty_env + sparse obstacles
        # D1: forest_env, D2: corridor_env, D3: docking_env, D4: dynamic_env
        environment_name = "empty_env"
        num_obstacles = 2  # task별로 조정
    
    class reward_config:
        goal_reach_bonus = 10.0
        collision_penalty = -10.0
        progress_coef = 1.0
        angular_velocity_coef = -0.01
```

각 다운스트림 task(D1~D4)는 이 config를 상속해 `environment_name`과 `num_obstacles` (또는 dynamic_env의 경우 `num_env_actions`)만 override한다. 이는 **4개 task를 단일 task class family로 통일**해 공정 비교를 코드 레벨에서 보장한다.

#### 4.9.2 관측 공간의 Raw Ray-cast 채택 (DCE encoder와의 차별점)

Aerial Gym의 공식 `navigation_task`는 default로 **Deep Collision Encoder (DCE)의 사전학습 latent**를 관측으로 사용한다. 이는 본 연구가 **의도적으로 사용하지 않는** 경로다. 이유:

- 본 연구의 핵심은 "backbone의 적응 전략 비교"이므로, 관측 단계는 가능한 한 raw하게 유지해야 혼란 변인이 적다.
- DCE latent는 이미 사전학습된 표현이어서, 그 위에 또 사전학습을 쌓으면 pretrain 기여도 분리가 어렵다.
- 32-d raw ray-cast는 크기가 작아 ~280K 정책에서 충분히 학습 가능.

**구현**: `navigation_task_config.py`의 `use_dce_encoder=False` 분기를 참조해 raw ray-cast 출력을 그대로 `observations["observations"]`에 채운다. Aerial Gym의 Warp 기반 ray-caster는 이미 GPU tensor로 depth를 반환하므로 추가 후처리 없이 정책 입력으로 사용한다.

#### 4.9.3 Sample Factory 통합 및 Recurrent Policy

Aerial Gym은 Sample Factory 통합 예제(`AerialGymVecEnv` wrapper)를 공식 제공한다. 본 연구는 이를 그대로 사용하되, 다음을 추가·수정한다.

**(a) Custom model class**: Sample Factory의 `encoder + core + decoder` 패턴을 따라 본 연구의 backbone을 정의.
- `Encoder`: proprio + extero MLP + fusion MLP (출력 128-d)
- `Core`: GRU 1-layer hidden=128 (Sample Factory의 `model_core_rnn` 설정 사용)
- `Decoder`: Actor head (Gaussian policy) + Critic head

Sample Factory의 `model_core_rnn` 설정은 GRU를 native 지원하며 hidden state 관리(episode reset 시 자동 zero-out)를 wrapper가 담당한다 — 저자는 모델 내부에서 state 처리를 구현할 필요 없음.

**(b) Action space와 controller interface**: Sample Factory actor head의 출력 4-d를 Aerial Gym의 `LeeRatesController`에 직접 전달. Aerial Gym이 이를 motor RPM으로 변환.

**(c) Parallel env 설정**: `num_envs=4096`, `rollout=32`, `batch_size=131072 (= 4096 × 32)`, `num_batches_per_epoch=8` → 실효 minibatch 16384 (§5.8과 일치).

#### 4.9.4 LoRA 구현 세부

**라이브러리 선택**: Hugging Face `peft` 라이브러리의 `LoraConfig` + `get_peft_model()` 패턴을 사용. `target_modules`로 Linear layer만 지정하면 PEFT가 자동으로 LoRA wrapping 수행.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,       # alpha/r = 2
    target_modules=[
        "fusion.0", "fusion.2",      # Fusion MLP의 2개 Linear
        "actor_head.0",              # Actor head 첫 Linear
        "critic_head.0",             # Critic head 첫 Linear
    ],
    lora_dropout=0.0,
    bias="none",
)
model_with_lora = get_peft_model(base_model, lora_config)
```

**Base model freezing**: `get_peft_model()` 호출이 base parameters를 자동 frozen 처리 (`requires_grad=False`). Encoder, GRU, 모든 backbone이 frozen.

**Sample Factory checkpoint 호환성**: Sample Factory는 state_dict 기반 checkpoint를 사용. 사전학습된 backbone state_dict를 로드한 후 LoRA config를 적용. PEFT 모델은 `save_pretrained()`로 LoRA adapter만 별도 저장 가능.

**Verification**: Training 시작 시 `model.print_trainable_parameters()`를 찍어 LoRA default에서 **~7K / ~280K (~2.5%)** 확인. 이 수치가 맞지 않으면 구조적 버그.

#### 4.9.5 Ablation A6 (LoRA + GRU)의 구현 경로

A6는 GRU gate matrices에도 LoRA를 주입한다. PEFT는 `nn.GRU`를 기본 지원하지 않으므로 두 경로 중 하나:

**경로 1 (primary)**: `nn.GRU`를 `nn.GRUCell` loop로 교체 + 각 gate weight에 수동 LoRA wrapping.
- 장점: 각 gate를 독립적으로 제어 가능, PEFT API로 처리 가능 (`nn.Linear` 취급)
- 단점: cudnn 가속 손실 → ~1.5x 추론 느려짐 (training 속도는 큰 영향 없음)
- 구현: ~200줄 PyTorch 추가 코드

**경로 2 (fallback)**: PEFT의 `_register_custom_module` experimental API 사용.
- 장점: `nn.GRU` 원형 유지, cudnn 가속 보존
- 단점: experimental API, 디버깅 어려움

본 연구는 경로 1을 default로 시도하되 구현 실패 시 A6만 negative report 처리. **main pipeline은 A6 성공 여부와 독립적**이므로 논문 완결성에 영향 없음.

#### 4.9.6 하드웨어 요구 검증

RTX 3090 (24GB VRAM) 환경에서의 리소스 추정:

| 항목 | 요구 | 여유 |
|------|------|------|
| Isaac Gym + Warp (4096 envs) | ~6 GB VRAM | 여유 |
| Policy (~280K) forward/backward | ~1 GB VRAM | 여유 |
| Rollout buffer (rollout=32, batch=131072) | ~2 GB VRAM | 여유 |
| Ray-cast rendering (32 rays × 4096 envs) | ~0.5 GB VRAM | 여유 |
| **총 VRAM 사용** | **~10 GB** | **~14 GB 여유** |
| 학습 1 run wall-clock | ~2.5 hours | — |

참고: DCE 공식 예제(더 큰 관측 + 4096 envs)가 RTX 3090에서 1시간에 학습 완료. 본 연구의 설정은 관측이 더 작고 학습 총량이 더 크므로 **2~3시간/run이 현실적 추정**.

#### 4.9.7 Dependencies 고정

재현성 및 4개월 일정 내 환경 호환 이슈 회피를 위해 다음 버전을 pin.

- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.8 (Isaac Gym 요구사항)
- **PyTorch**: 2.0.1 + cu118
- **Isaac Gym**: Preview 4 (Aerial Gym 요구사항)
- **Aerial Gym**: v2.0.0 (github release tag 고정)
- **Sample Factory**: 2.1.x
- **PEFT**: 0.7.x

`environment.yml` / `requirements.txt`를 proposal 저장소에 포함해 즉시 재현 가능하게 함.

#### 4.9.8 구현 가능성 최종 검증

본 §4.9의 명세가 모든 핵심 질문에 답한다.

| 질문 | 답 |
|------|-----|
| 관측 공간 45-d를 Aerial Gym에서 구성 가능한가? | ✓ raw ray-cast 32 + proprio 13 조합, task config에서 명시 |
| `base_quad` + `LeeRatesController`로 CTBR 4-d 출력 가능한가? | ✓ Aerial Gym 표준 조합, 공식 예제 존재 |
| Sample Factory recurrent PPO + custom model이 가능한가? | ✓ Sample Factory native 지원, DCE 예제로 검증됨 |
| PEFT LoRA가 Sample Factory 모델과 호환되는가? | ✓ `nn.Linear` target_modules 지정으로 표준 적용 |
| 사전학습 checkpoint → 세 적응 전략 각각으로 초기화? | ✓ state_dict load + (FullFT: as-is / Frozen: requires_grad 조정 / LoRA: get_peft_model 적용) |
| 4개 downstream task가 코드 수준으로 통일 구조인가? | ✓ 단일 `PretrainAdaptTaskConfig` 상속, env_name만 override |
| RTX 3090에서 resource 여유 있나? | ✓ ~10GB VRAM 사용 (24GB 중), 2.5h/run 추정 |
| A6 (GRU LoRA) 실패해도 main이 완결되는가? | ✓ A6는 main과 독립, fallback 경로 존재 |

---

## 5. Experimental Setup

### 5.1 Research Questions

- **RQ1 (Paradigm Validation)**: 사전학습-적응 패러다임은 동일 총 예산의 scratch PPO 대비 성능·효율·안정성을 개선하는가?
- **RQ2 (Three-way Comparison)**: Full-FT / Encoder-Frozen / LoRA 세 전략은 다섯 축에서 어떻게 다른가?
- **RQ3 (Task-Dependence)**: 세 전략의 상대적 우위는 다운스트림 과업 특성(환경 복잡도, 외란 수준, 시간 구조)에 따라 어떻게 변하는가?
- **RQ4 (LoRA Internal)**: LoRA의 rank, 주입 위치, GRU 적응 여부는 성능에 어떤 영향을 주는가?

### 5.2 Simulation Environment

**시뮬레이터**: Aerial Gym Simulator v2.0.0, **Warp 백엔드** (GPU ray-casting).
**병렬 환경**: 2048~4096 parallel envs (Aerial Gym의 대규모 병렬성 최대 활용).
**RL 라이브러리**: Sample Factory (Aerial Gym 공식 통합, recurrent PPO 안정 지원, DCE 논문과 동일 stack).
**기체**: `base_quad` (Aerial Gym 표준 under-actuated quadrotor).
**제어 인터페이스**: `LeeRatesController` 위의 CTBR command, 50 Hz.
**관측**: Aerial Gym의 Warp 기반 ray-cast (32 rays, 수평 360°, 11.25° 분할).

### 5.3 Tasks Mapped to Aerial Gym Environments

Aerial Gym의 실제 environment 모듈을 활용하여 task diversity 확보.

**사전학습 $\mathcal{T}^{\text{pre}}$ — empty + sparse goal reaching**
- Environment: `empty_env` + 확률적 정적 장애물 1~3개
- Reward: $r_t = -\Delta\|p - g\|_2 + 10 \cdot \mathbb{1}[\text{goal}] - 10 \cdot \mathbb{1}[\text{collision}]$
- 목적: 기본 비행 제어 + goal-seeking + 약한 장애물 인식

**다운스트림 과업 (4개)** — Aerial Gym의 사전 구현된 environment를 최대한 활용하여 재현성·fair comparison 확보.

| Task | Aerial Gym Env | Description | Distance from Pretrain |
|------|----------------|-------------|----------------------|
| **D1: Dense Forest** | `forest_env` | 정적 원기둥(나무) 조밀 분포, dense obstacle navigation | In-distribution 확장 |
| **D2: Corridor Navigation** | `corridor_env` | 긴 복도 + 부분적 개구부 통과, 공간 구조 다름 | OOD 공간 구조 |
| **D3: Precision Docking** | `docking_env` | 정밀 위치 고정 + 각도 제어, 시간 구조 다름 | OOD 과업 유형 |
| **D4: Dynamic Obstacles** | `dynamic_env` | 이동하는 장애물 5개 (num_env_actions=6), 시간 의존성 높음 | OOD 동역학 |

**이 4개 과업이 실험적으로 우수한 이유**:
- 모두 Aerial Gym에 **사전 구현되어 있어** 환경 신뢰성·재현성 확보
- 사전학습 분포와의 거리 스펙트럼 형성 (D1 가까움 → D4 가장 멀음)
- 각기 다른 도전 — 정적 밀도(D1), 공간 구조(D2), 과업 유형(D3), 동역학(D4)
- RQ3(과업 특성 의존성)에 대한 풍부한 실증적 답 가능

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
3. **B-Frozen**: Pretrained + Encoder-Frozen (DCE-style)
4. **Ours-LoRA-r8** (default): Pretrained + Linear-only LoRA rank 8

**Ablations (3 seeds)**:
- A1 — LoRA rank: $r \in \{1, 2, 4, 16\}$
- A2 — LoRA injection location: {fusion-only, heads-only, both}
- A3 — Pretraining objective: with/without $\mathcal{L}_{\text{dyn}}$
- A4 — Actor-only LoRA (vs Actor+Critic default)
- A5 — LoRA learning rate: $\{1\text{e-}3, 3\text{e-}3\}$
- **A6 — LoRA + GRU**: GRU gate matrices에도 LoRA 주입 (D3, D4 한정, custom wrapper)
- **A7 — Robustness evaluation**: main 4 조건을 4가지 variant 환경에서 zero-shot 평가

### 5.6 Dependent Variables (5 Axes + Robustness)

**축 1 — Performance**: Final success rate, collision rate, time-to-goal, cumulative return

**축 2 — Sample efficiency**: Steps-to-threshold (success rate 0.8 도달), learning curve AUC

**축 3 — Convergence stability**: Final performance의 seed std/IQR (5 seeds), 학습 발산 횟수

**축 4 — Hyperparameter sensitivity**: lr × entropy 3×3 sweep 결과의 IQR

**축 5 — Forgetting magnitude**: 적응 후 정책을 $\mathcal{T}^{\text{pre}}$에서 재평가한 retention rate. $\text{Forgetting} = \text{perf}_{\text{pre, before}} - \text{perf}_{\text{pre, after}}$

**보조 축 — Robustness (A7)**: 4개 variant 환경에서의 성능 drop.

**Resource metrics**: Trainable param 수, 학습 wall-clock time, 추론 latency (RTX 3090 batch=1), memory footprint.

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

**Success**: 목표 위치 0.5m 이내 진입 + 3초 dwell.
**Failure**: 충돌 (0.2m), 고도 위반 ($0.5 < z < 5$), timeout.
**Reward (모든 task 공통 구조)**:
$$r_t = -\Delta\|p_t - g\|_2 + 10 \cdot \mathbb{1}[\text{success}] - 10 \cdot \mathbb{1}[\text{collision}] - 0.01 \|\omega_t\|_2^2$$

### 5.10 Run Count and Budget

| 실험 | Conditions × Tasks × Seeds | Runs |
|------|-----------------------------|------|
| Main | 4 × 4 × 5 | 80 |
| A1 (rank) | 4 × 4 × 3 | 48 |
| A2 (location) | 3 × 2 × 3 | 18 |
| A3 (dyn loss) | 1 × 4 × 3 | 12 |
| A4 (actor-only) | 1 × 4 × 3 | 12 |
| A5 (higher lr) | 2 × 2 × 3 | 12 |
| A6 (LoRA+GRU) | 1 × 2 × 3 | 6 |
| A7 (robustness) | 4 × 4 × 4 variants (eval only) | — |
| Hyperparameter sweep | 9 × 4 × 1 | 36 |
| **Total training runs** | | **~224** |

**GPU budget**: 평균 2.5h/run × 224 ≈ **560 GPU-hours** + buffer, RTX 3090 24/7 기준 약 24일. 4개월 일정에 여유.

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

- **H1 (RQ1)**: 사전학습-적응 세 조건 모두 Scratch 대비 steps-to-threshold 30~50% 단축, seed std 50%+ 감소.
- **H2 (RQ2)**: Full-FT가 D1에서 최고 성능 but 가장 높은 forgetting; Frozen은 D1, D2에서 견고 but D3에서 천장; LoRA-r8이 ~2.5% trainable로 Full-FT 90%+ 성능 회복, forgetting 가장 적음.
- **H3 (RQ3)**: D1→D4로 갈수록 Frozen의 단점이 노출, Full-FT/LoRA의 우위 커짐. LoRA는 항상 Pareto-optimal에 가까움.
- **H4 (RQ4)**: Rank 2~8이 sweet spot. Linear-only LoRA가 대부분 task에서 Full-FT 90%+ 성능. A6에서 GRU LoRA는 D3, D4에 한정해 marginal 개선.
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

### Contribution 1 — Paradigm Transplantation

NLP/CV의 pretrain-then-adapt 패러다임을 UAV 항법 RL에 명시적으로 이식하고, **총 환경 상호작용 예산 통제 하의 공정 비교**로 유효성을 정량 입증.

### Contribution 2 — First Fair Three-way Comparison in UAV

Full-FT, Encoder-Frozen, LoRA를 동일 backbone·동일 환경·동일 예산에서 비교한 첫 UAV RL 연구. 다섯 축 평가로 trade-off 곡선을 명시적 매핑.

### Contribution 3 — Empirical Guidelines for Strategy Selection

다운스트림 과업의 (i) 사전학습 분포 유사도, (ii) 환경 복잡도, (iii) 시간 구조 변화, (iv) 외란 수준에 따른 전략 선택 empirical map 제공. 실무자 의사결정 근거.

### Contribution 4 — Reproducible Aerial Gym Protocol

- Sample Factory + Aerial Gym 기반 공정 비교 reference implementation
- 224+ runs raw 학습 곡선
- Linear-only LoRA injection 표준 패턴
- (Optional) GRU-LoRA wrapper (A6 성공 시)
- GitHub 공개 (paper acceptance 후)

---

## 8. Execution Plan and Risk Management

### 8.1 Timeline (4 months post thesis)

| 월 | 작업 | 결정 게이트 |
|----|------|-------------|
| **M1** | Aerial Gym + Sample Factory 환경 검증, backbone 구현, 사전학습 파이프라인, 1 seed로 4 conditions × D1 smoke test | 4 main condition 모두 수렴 확인 |
| **M2** | 4 tasks (`forest_env`, `corridor_env`, `docking_env`, `dynamic_env`) 연결, main 실험 시작 (80 runs) | Main 50% 완료 |
| **M3** | Main 완료, ablation A1~A5 (~102 runs), A7 robustness eval, 선택적으로 A6 시도 | 모든 raw data 확보 |
| **M4** | 결과 분석, figures/tables 작성, 논문 초고, 내부 검토, Drones 제출 | — |

### 8.2 Main Risks and Mitigations

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| Recurrent PPO 학습 불안정 | 중 | 높음 | Sample Factory 검증 hyperparameter, 필요 시 GRU → frame stack fallback |
| LoRA + GRU (A6) 구현 실패 | 중 | **낮음** | Main 결과에 영향 없음. A6만 negative report 가능 |
| LoRA가 Full-FT 대비 크게 떨어짐 | 낮음 | 중 | Rank 확대 또는 주입 위치 확장. Finding 자체로 "UAV RL에서 LoRA 한계" 기여 |
| Reviewer "GRU LoRA 왜 안 하냐" 지적 | 중 | 중 | A6로 직접 답변, §4.7에서 근거 사전 제시 |
| Aerial Gym environment 호환성 | 낮음 | 중 | 공식 release tag 고정, conda env 스냅샷 |
| 모든 전략 비슷한 성능 | 중 | 중 | "선택이 robust"라는 finding도 기여 |
| 4개월 내 ablation 미완 | 낮음 | 낮음 | Main 우선, A2/A5 축소 가능 |
| Drones reviewer가 sim-only 비판 | 낮~중 | 낮음 | Future work에 sim2real 명시 |

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
- 4개가 Aerial Gym의 잘 정의된 environments (`forest_env`, `corridor_env`, `docking_env`, `dynamic_env`)에 자연스럽게 매핑
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

- **Within scope**: 회전익 navigation, 외란/노이즈 robustness, 동적 장애물, 대규모 병렬 RL
- **Out of scope**: 고정익/VTOL (Aerial Gym 미지원), 다기체/편대 (미지원), 공간적 wind field (미지원, 균일 외란으로 근사)

본 연구의 scope는 Aerial Gym이 가장 강점을 보이는 영역에 정확히 부합한다.

### A.8 왜 flat 정책인가 (계층적 추상화 vs. 본 연구의 선택)

본 연구는 **flat 정책**을 사용한다 — 관측에서 CTBR action까지 하나의 신경망이 직접 매핑. 대안으로 **계층적 추상화**(상위 신경망이 3D waypoint 또는 속도 명령을 생성, 하위 신경망이 CTBR로 변환)가 있다. 후자는 본질적으로 **월드모델의 잠재공간** 아이디어와 유사하다 — RL이 다뤄야 할 행동 공간 추상화 수준을 높이면 학습 난이도가 낮아진다.

계층적 추상화는 매력적이지만 본 연구에는 직접 도입하지 않았다.

1. **Scope 확산 위험**: 계층 분해를 도입하면 "어떤 수준에서 분해할 것인가", "상위/하위 경계를 어떻게 설정할 것인가" 같은 별개의 설계 질문이 논문 중심이 되어, 본 연구의 핵심 질문(3-way 적응 전략 비교)이 흐려진다.
2. **공정 비교 어려움**: 계층 구조를 도입하면 Full-FT/Frozen/LoRA가 "어느 수준에서 적용되는지"가 조건별로 달라져 비교의 명확성이 떨어진다.
3. **보완 관계**: 계층적 추상화와 본 연구의 pretrain-adapt 패러다임은 **상호 배타적이지 않다**. 계층 구조 위에서도 pretrain-adapt가 적용 가능하며, 오히려 후속 연구의 자연스러운 확장 방향이다.

따라서 계층적 추상화는 **본 연구의 scope에서 제외하되, future work로 명시**한다 (Appendix B 참조). DCE가 이미 supervised pretrained depth encoder + RL policy라는 "얕은 계층 분해"를 사용하므로, 본 연구의 backbone 자체가 부분적으로 이 아이디어를 이미 반영하고 있기도 하다.

### A.9 Assumptions That May Need Revision Later

- $B_{\text{total}} = 2 \times 10^8$은 DCE의 실험 규모에서 추정. 실제 smoke test 후 조정 필요할 수 있음.
- LoRA lr $3 \times 10^{-4}$은 Hu et al. 2021 관례. RL 맥락에서 재튜닝 여지.
- GRU hidden=128은 DCE의 64보다 큼. 학습 불안정성 발생 시 64로 축소 고려.

---

## Appendix B. Known Limitations

- **Sim-only**: 실제 UAV에서 동일한 trade-off 관찰 미보장, LoRA의 UAV sim2real은 미검증
- **단일 backbone, 단일 알고리즘**: ~280K recurrent + PPO 조합에 한정. 큰 backbone이나 off-policy에서는 다를 수 있음
- **Flat 정책에 한정**: 본 연구는 관측 → CTBR action을 직접 매핑하는 flat 정책만 비교한다. 상위 신경망이 waypoint·속도 같은 고수준 명령을 생성하고 하위 신경망이 이를 실제 제어 명령으로 변환하는 **계층적 추상화** (월드모델의 잠재공간 개념과 유사)는 본 연구의 3-way 적응 비교와 **상호 보완적** 방향이며, 본 연구의 자연스러운 확장으로 future work에서 다룰 가치가 있다.
- **LoRA의 RL-specific 이론 부재**: LoRA는 supervised loss 가정, RL의 비정상 reward·exploration과의 상호작용 이론 분석은 scope 외
- **Forgetting 측정의 한계**: 사전학습 환경에서의 retention은 first approximation, Wołczyk et al. 2024의 정교한 protocol 미적용
- **4-task 일반화**: 4개 task로 trade-off 일반화는 위험할 수 있음. 5+ task 확장은 future work
- **Aerial Gym 자체의 scope 제한**: 고정익, VTOL, 다기체 실험은 이 시뮬레이터에서 본질적으로 불가. 다른 시뮬레이터(e.g., AirSim, Flightmare) 사용 시 별도 연구

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

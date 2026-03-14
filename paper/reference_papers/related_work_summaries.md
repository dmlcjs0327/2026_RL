# Related Work 논문 요약

본 문서는 **Continuation-Derived Relational Initialization for Goal-Conditioned Critics in Sparse-Reward Aerial Reinforcement Learning** (T-RO 목표) 논문의 related work에 참고할 논문들을, 각 PDF 내용을 바탕으로 정리한 것이다.  
각 논문마다 ① 논문명, ② 다루는 문제, ③ 제안 솔루션, ④ 실험, ⑤ 주요 기여, ⑥ 본 논문과의 연관성을 기술한다.

---

## Group A: Sparse-Reward Goal-Conditioned RL (기초, relabeling, exploration, reformulation)

### 1. Universal Value Function Approximators (Schaul et al., 2015)

- **논문명:** Universal Value Function Approximators (UVFA)
- **다루는 문제:** 단일 목표가 아닌 여러 목표에 대해 하나의 value function으로 일반화하는 문제. 목표 공간과 상태 공간 모두에서 구조를 활용해 효율적으로 학습해야 함.
- **제안 솔루션:** 상태와 목표를 각각 임베딩 φ(s), ψ(g)로 매핑하고, 이들의 조합(예: 내적)으로 V(s,g) 또는 Q(s,a,g)를 근사. 행렬 분해 기반 2단계 학습으로 UVFA를 학습하고, RL에서는 bootstrapping으로 직접 보상만으로 학습.
- **실험:** LavaWorld 등에서 supervised UVFA 학습 속도·일반화, Ms Pacman에서 이전에 본 적 없는 목표(펠릿)에 대한 일반화.
- **주요 기여:** 목표까지 포함한 universal value function 근사 제안; 상태·목표 공간 구조를 동시에 활용하는 학습 방법; HER 등 goal-conditioned RL의 함수 근사 기반.
- **본 논문과의 연관성:** 본 논문의 goal-conditioned critic Q(s,a,g) 및 shared state-goal encoder f_θ(s,g)의 이론적 뿌리. “같은 context 아래 여러 목표에 대한 상대적 선호”를 다루는 본 논문의 문제 설정은 UVFA가 전제하는 goal-conditioned value 구조 위에 있음.

---

### 2. Hindsight Experience Replay (Andrychowicz et al., 2017)

- **논문명:** Hindsight Experience Replay (HER)
- **다루는 문제:** Sparse·binary reward만 있을 때 샘플 효율이 낮고 학습이 거의 불가능한 문제. 실패 궤적에서도 유용한 학습 신호를 얻어야 함.
- **제안 솔루션:** 에피소드 종료 후, 원래 목표 대신 “실제로 도달한 상태”(achieved goal)를 목표로 재해석해 replay에 추가. UVFA + off-policy 알고리즘(DQN, DDPG 등)과 결합. 암묵적 curriculum으로 작동.
- **실험:** Bit-flipping, Fetch 로봇 manipulation(pushing, sliding, pick-and-place); binary reward만 사용, 물리 로봇 배치.
- **주요 기여:** Sparse reward만으로 goal-conditioned RL 학습 가능하게 함; 실패를 성공으로 재해석하는 relabeling; 본 논문에서도 RL backbone으로 HER+SAC를 사용하는 직접적 기반.
- **본 논문과의 연관성:** 본 논문은 HER를 유지한 채 “초기 구간에서 critic의 same-context goal ordering”만 개선하는 초기화를 제안. HER가 만드는 relabeled data와 continuation에서 얻은 relational evidence를 구분해, 후자로 critic encoder를 warm-start하는 위치를 명확히 함.

---

### 3. Curriculum-Guided Hindsight Experience Replay (Fang et al., 2019)

- **논문명:** Curriculum-Guided HER (CHER)
- **다루는 문제:** HER가 모든 실패 궤적을 균등하게 replay하는데, 학습 단계에 따라 유용한 hindsight가 다름. 목표 근접성과 다양성(호기심)의 균형이 필요.
- **제안 솔루션:** Goal-proximity(목표와의 유사도)와 diversity-based curiosity(시설 위치 함수 등 submodular)를 결합한 curriculum으로 hindsight에서 재생할 목표 집합을 선택. 학습이 진행됨에 따라 curiosity 비중을 줄이고 proximity 비중을 키움.
- **실험:** Fetch 등 로봇 manipulation; CHER가 HER보다 학습 효율·최종 성능에서 우수.
- **주요 기여:** HER에 curriculum을 도입해 “어떤 hindsight를 얼마나 재생할지”를 조절; goal-proximity와 diversity의 스케줄링.
- **본 논문과의 연관성:** 본 논문은 curriculum이 아니라 “같은 context 내 candidate goal 간 relational ordering”으로 critic을 warm-start함. CHER는 replay 선택, 본 논문은 critic encoder 초기화라는 서로 다른 레이어를 다룸.

---

### 4. Exploration via Hindsight Goal Generation (Ren et al., 2019)

- **논문명:** Hindsight Goal Generation (HGG)
- **다루는 문제:** HER의 hindsight goal이 원하는 목표 분포와 멀어져 exploration이 비효율적인 경우. 단기적으로 달성 가능하면서 장기적으로 실제 목표에 도움이 되는 중간 목표가 필요.
- **제안 솔루션:** 가치 함수의 Lipschitz 연속성 가정 하에, 실제 목표 분포와의 Wasserstein 거리를 이용한 surrogate 목표 분포 최적화. Discrete Wasserstein Barycenter solver로 hindsight goal 집합 생성 후 exploration에 사용.
- **실험:** Fetch 등 manipulation; DDPG+HER 대비 샘플 효율 향상, ablation으로 하이퍼파라미터 견고성.
- **주요 기여:** Hindsight goal을 휴리스틱이 아닌 최적화로 생성; 목표 분포와의 거리 기반 이론적 바운드.
- **본 논문과의 연관성:** 본 논문은 “어떤 goal을 선택할지”보다 “주어진 candidate goals 간 same-context ordering을 critic이 얼마나 잘 배우는지”에 초점. HGG는 goal 생성, 본 논문은 critic의 ordering 품질 개선.

---

### 5. Learning to Reach Goals via Iterated Supervised Learning (Ghosh et al., 2021, GCSL)

- **논문명:** Learning to Reach Goals via Iterated Supervised Learning (GCSL)
- **다루는 문제:** Sparse-reward goal-reaching에서 value function 기반 RL이 불안정하고 하이퍼파라미터에 민감한 문제. 전문가 시연 없이 안정적으로 목표 도달 정책을 학습하고 싶음.
- **제안 솔루션:** Hindsight relabeling으로 궤적의 최종 상태를 목표로 삼아 “그 목표에 대한 전문가 시연”으로 재해석하고, 행동 cloning으로 목표 조건 정책 학습. 반복적으로 데이터 수집·relabel·imitation. Value function 없이 RL 목표의 하한을 최적화함을 정리로 제시.
- **실험:** 로봇 manipulation 벤치마크; value 기반/정책 경사 대비 안정성·성능.
- **주요 기여:** Goal-conditioned RL을 value 없이 순수 supervised learning으로; hindsight relabeling과 imitation의 결합 및 이론적 연결.
- **본 논문과의 연관성:** 본 논문은 value/critic을 유지하고 그 encoder만 warm-start함. GCSL은 value 제거, 본 논문은 critic 구조를 살리면서 초기 ordering만 보완한다는 점에서 대비됨.

---

### 6. C-Learning: Learning to Achieve Goals via Recursive Classification (Eysenbach et al., 2021)

- **논문명:** C-Learning: Learning to Achieve Goals via Recursive Classification
- **다루는 문제:** 연속 공간에서 sparse reward 1(s=g)에 대한 Q-learning은 확률이 0이 되어 의미가 없음. Goal-conditioned RL을 “미래 상태 분포 예측/제어”로 재정의할 필요.
- **제안 솔루션:** 미래 상태를 “미래 vs 무작위 상태” 이진 분류로 구분하는 classifier 학습 → 베이즈 규칙으로 미래 상태 밀도 추정. Bootstrapping으로 다른 정책의 미래 분포도 추정 가능하게 하고, 이를 goal-conditioned RL 알고리즘으로 연결. 목표 샘플링 비율에 대한 가설을 실험으로 검증.
- **실험:** 로봇 시뮬레이션 goal-reaching; 밀도 추정 정확도, 기존 goal-conditioned 방법과의 성능 비교.
- **주요 기여:** Goal-conditioned RL을 밀도 추정으로 재구성; Q-learning 한계(연속 공간·sparse reward) 명시 및 classifier 기반 대안; 목표 샘플링 비율에 대한 통찰.
- **본 논문과의 연관성:** 본 논문은 Q/critic을 “값 회귀”가 아니라 “same-context relational ordering”에 쓰는 초기화로만 사용. C-learning의 밀도 관점과 본 논문의 ordering 관점은 보완적; continuation에서 나온 evidence를 ordering에만 쓰는 본 논문의 제한적 해석과도 정합적.

---

## Group B: Goal-Conditioned Representation Learning, Future-Derived Supervision, Goal Modeling

### 7. Contextual Imagined Goals for Self-Supervised Robotic Learning (Nair et al., 2020, CIG)

- **논문명:** Contextual Imagined Goals for Self-Supervised Robotic Learning (CIG)
- **다루는 문제:** 장면·물체가 바뀌는 환경에서 self-supervised goal-conditioned RL 시, 과거 경험이나 단일 생성 모델로 제안한 목표가 “현재 맥락에서 달성 불가능”한 경우가 많음.
- **제안 솔루션:** Context-conditioned VAE(CC-VAE): 맥락(초기 상태 이미지)은 정보를 자유롭게 쓰고, 변화 가능한 부분만 bottleneck으로 두어 “현재 맥락에서 도달 가능한” 목표만 생성. 이렇게 만든 latent goal으로 off-policy goal-conditioned RL(CC-RIG).
- **실험:** 시각적으로 다양한 pusher, 실제 Sawyer 로봇; 훈련 시 본 적 없는 물체에 대한 일반화.
- **주요 기여:** 맥락 조건 목표 생성으로 달성 가능한 목표만 제안; 시각적 다양성 있는 환경에서 self-supervised GC-RL.
- **본 논문과의 연관성:** “같은 context에서 어떤 goal이 더 feasible한가”라는 질문과 맞닿음. 본 논문은 context-conditioned goal 생성이 아니라, 주어진 candidate goals에 대해 critic의 same-context ordering을 continuation 기반으로 warm-start한다는 점에서 구분됨.

---

### 8. Discovering and Achieving Goals via World Models (Mendonça et al., 2021, LEXA)

- **논문명:** Discovering and Achieving Goals via World Models (LEXA)
- **다루는 문제:** 비지도로 다양한 목표 도달을 학습할 때, 기존 방법은 “이미 방문한 상태”나 생성 모델로 목표를 제안해 frontier 밖 탐험이 부족함.
- **제안 솔루션:** World model(RSSM)로 상상 궤적 생성. Explorer는 model 내에서 disagreement 등으로 “놀라운” 미래 상태를 계획해 실행하고, Achiever는 그렇게 발견된 상태를 목표로 삼아 상상 궤적에서 목표 도달 학습. Foresight 기반 탐험.
- **실험:** RoboYoga, RoboBins, RoboKitchen 등 40개 목표 이미지 벤치마크; Kitchen에서 최초 성공, 다물체 순차 조작.
- **주요 기여:** World model + explorer/achiever 분리; foresight 기반 목표 발견; zero-shot goal image 도달.
- **본 논문과의 연관성:** continuation에서 얻은 “제한된 relational evidence”와 유사하게, world model 상상 궤적에서 나온 신호를 쓰는 점에서 연결. 본 논문은 value regression이 아닌 pairwise ordering warm-start만 추출한다는 점을 대비해 기술할 수 있음.

---

### 9. Variational Empowerment as Representation Learning for Goal-Conditioned RL (Choi et al., 2021, VGCRL)

- **논문명:** Variational Empowerment as Representation Learning for Goal-Conditioned RL (VGCRL)
- **다루는 문제:** GCRL과 MI/empowerment 기반 skill discovery를 통합된 관점에서 보고, 목표 공간·표현·평가를 정리할 필요.
- **제안 솔루션:** 표준 GCRL 목표를 variational MI의 특수 케이스로 해석. 적응 분산·선형 매핑 GCRL 변형, spectral normalization으로 latent goal 품질 개선, HER를 MI 목표에 적용한 P-HER, latent goal reaching(LGR) 메트릭 제안.
- **실험:** Empowerment/GCRL 변형 비교, LGR로 skill 품질 평가.
- **주요 기여:** GCRL과 MI 기반 RL의 수학적 통합; discriminator/표현 정규화가 목표 도달에 미치는 영향 분석; HER의 MI 쪽 적용.
- **본 논문과의 연관성:** Critic/value 쪽 표현과 goal ordering을 다루는 본 논문과, empowerment·skill·목표 공간을 다루는 VGCRL은 같은 “goal-conditioned representation” 계열. 본 논문은 HER를 유지한 채 critic encoder만 relational warm-start한다는 차이를 강조할 수 있음.

---

### 10. Contrastive Learning as Goal-Conditioned Reinforcement Learning (Eysenbach et al., 2022)

- **논문명:** Contrastive Learning as Goal-Conditioned Reinforcement Learning
- **다루는 문제:** RL에서 표현 학습을 위해 보조 손실·데이터 증강을 쓰는 방식이 많음. “표현 학습 알고리즘 자체가 goal-conditioned RL이 되게” 할 수 있는지.
- **제안 솔루션:** (s,a)와 미래 상태를 positive로, 무작위 상태를 negative로 하는 contrastive 학습. Discounted state occupancy로 positive를 샘플링하면, 학습된 내적이 특정 보상에 대한 Q와 동치임을 증명. C-learning 일반화이자 더 단순한 방법 제안; offline에서도 적용.
- **실험:** Goal-conditioned 벤치마크, 이미지 관측, offline 설정; 데이터 증강·보조 목표 없이 기존 방법 대비 성능.
- **주요 기여:** Contrastive 학습과 goal-conditioned value의 동치 관계; single objective로 표현·RL 동시 학습; C-learning의 contrastive 해석.
- **본 논문과의 연관성:** 본 논문의 baseline으로 “generic SSL / contrastive”가 등장. 본 논문은 contrastive가 아닌 same-context pairwise ranking으로 critic encoder만 warm-start하고, temporal-only·generic SSL과 구별되는 실험 설계로 차별점을 둠.

---

## Group C: Value-Aware / Critic-Side Representation and Auxiliary Learning

### 11. Return-Based Contrastive Representation Learning for RL (Liu et al., 2021, RCRL)

- **논문명:** Return-Based Contrastive Representation Learning for Reinforcement Learning (RCRL)
- **다루는 문제:** 보조 과제가 RL에 도움이 되지만, return 같은 RL 고유 신호를 쓰는 contrastive 보조 과제는 부족함.
- **제안 솔루션:** Anchor (s,a)에 대해 같은/비슷한 return을 가진 것을 positive, 다른 return을 negative로 두고 contrastive 보조 손실. Z^π-irrelevance라는 state-action abstraction과 연결하고, 이 추상화가 Q 근사에 유리함을 이론적으로 제시.
- **실험:** Atari, DeepMind Control, low data regime; Rainbow/SAC와 결합, 기존 contrastive 보조와 병합 시 추가 이득.
- **주요 기여:** Return 기반 contrastive 보조 과제; 추상화 이론과의 연결; critic/값과 맞는 표현 학습.
- **본 논문과의 연관성:** 본 논문도 critic 쪽 표현을 다루지만, return이 아니라 “same-context goal 간 pairwise ordering”으로 warm-start. RCRL은 보조 과제, 본 논문은 초기화 단계만 사용한다는 차이가 있음.

---

### 12. On the Effect of Auxiliary Tasks on Representation Dynamics (Lyle et al., 2021)

- **논문명:** On the Effect of Auxiliary Tasks on Representation Dynamics
- **다루는 문제:** 보조 과제가 표현을 어떻게 바꾸는지, 어떤 보조 과제가 sparse reward에 유리한지 이해 부족.
- **제안 솔루션:** TD·Monte Carlo 학습 역학을 이론적으로 분석; 전이 연산자의 스펙트럼 분해와 보조 과제가 유도하는 표현 부분공간을 연결. Random cumulant 보조 과제가 sparse reward에 특히 유리하다는 가설을 ALE에서 검증.
- **실험:** 소규모 MDP에서 역학 분석, ALE에서 random cumulant 등 보조 과제 비교.
- **주요 기여:** 보조 과제 → 표현 역학의 이론적 연결; sparse reward에서 random cumulant의 효과.
- **본 논문과의 연관성:** 본 논문은 보조 과제가 아니라 “continuation-derived relational” 한 종류의 신호로 critic encoder만 warm-start. 보조 과제로 인한 표현 변화에 대한 Lyle et al.의 관점을 인용해, 본 논문의 초기화가 critic 학습 역학에 미치는 영향을 논의할 수 있음.

---

### 13. Value-Consistent Representation Learning for Data-Efficient RL (Yue et al., 2023, VCR)

- **논문명:** Value-Consistent Representation Learning for Data-Efficient Reinforcement Learning (VCR)
- **다루는 문제:** SPR 등 transition 예측 기반 표현은 “인식”에는 좋지만, 결정에 필요한 value 추정과는 어긋날 수 있음.
- **제안 솔루션:** Dynamics model로 imagined 궤적을 만든 뒤, 실제 상태와 imagined 상태에서 나온 Q 분포를 맞추는 value-consistent 손실. 보상은 실제 환경만 사용. SPR + value consistency로 search-free RL에서 SOTA.
- **실험:** Atari 100K, DeepMind Control; Q error, 샘플 효율.
- **주요 기여:** Value equivalence를 action-value에 적용; transition 학습에 value 일관성 추가.
- **본 논문과의 연관성:** 본 논문도 “critic/value에 맞는 표현”을 지향하지만, VCR은 transition 모델 + value 일관성이고, 본 논문은 continuation 기반 pairwise ordering으로 critic encoder만 warm-start. “Value에 맞는 표현”이라는 목표는 공유하나 방법이 다름.

---

## Group D: Aerial Robotics, Local Feasibility, Approach-Sensitive Agile Flight

### 14. Aggressive Quadrotor Flight through Narrow Gaps with Onboard Sensing (Falanga et al., 2017)

- **논문명:** Aggressive Quadrotor Flight through Narrow Gaps with Onboard Sensing and Computing Using Active Vision
- **다루는 문제:** 좁은 gap 통과 시 외부 위치 추정이나 gap 사전 정보 없이, 온보드 비전·IMU만으로 기하·동역학·인식 제약을 만족하는 궤적 생성 및 실행.
- **제안 솔루션:** Gap 통과 구간은 gap 중심·평면 제약으로 궤적 설계; 접근 구간은 카메라가 gap을 계속 보도록 perception 제약을 넣고, 불확실성이 거리 제곱으로 커지는 것을 고려해 실행 중 재계획. Active vision과 궤적 생성 통합.
- **실험:** 실제 쿼드로터, 45° 기울어진 gap, 80% 성공률.
- **주요 기여:** 온보드만으로 aggressive narrow-gap 통과; 기하·동역학·인식 제약 통합; 재계획으로 불확실성 대응.
- **본 논문과의 연관성:** “Terminal position alone으로는 부족하고, 진입 각·자세가 중요한 aerial 과제”의 대표 사례. 본 논문의 Task 3/4(corridor, docking)와 동일한 맥락에서 인용 가능.

---

### 15. Champion-Level Drone Racing Using Deep RL (Kaufmann et al., 2023, Swift)

- **논문명:** Champion-Level Drone Racing Using Deep Reinforcement Learning (Swift)
- **다루는 문제:** FPV 드론 레이싱에서 월드 챔피언 수준의 자율 비행을 온보드 센서만으로 달성. 시뮬-실제 차이(역학·인식) 보정 필요.
- **제안 솔루션:** 관측 정책(VIO + gate 검출, Kalman filter)으로 저차원 상태 추정, 제어 정책(MLP)은 시뮬레이션에서 model-free on-policy RL로 학습. 실제 데이터로 perception·dynamics residual을 Gaussian process·k-NN 등으로 모델링해 시뮬에 반영 후 fine-tune.
- **실험:** 실제 레이스 트랙, 인간 챔피언과 대결, Swift가 여러 경기 승리 및 최단 랩타임.
- **주요 기여:** 온보드만으로 챔피언 수준 드론 레이싱; 시뮬-실제 residual 모델링으로 전이; T-RO/Nature 수준 벤치마크.
- **본 논문과의 연관성:** Agile aerial RL의 최신 성과. 본 논문은 레이싱 자체가 아니라 sparse-reward GC-RL의 critic 초기화이므로, “agile flight·드론 RL” 맥락과 “우리는 critic warm-start”라는 차이를 related work에서 명시할 수 있음.

---

### 16. Learning Perception-Aware Agile Flight in Cluttered Environments (Song et al., 2023)

- **논문명:** Learning Perception-Aware Agile Flight in Cluttered Environments
- **다루는 문제:** 최소 시간 비행에서 카메라 시야 제약(perception-aware)을 고려하지 않으면, 목표 방향이 안 보이거나 반응이 늦어짐.
- **제안 솔루션:** Privileged learning: full-state로 minimum-time + obstacle avoidance + perception-aware 보상으로 teacher RL 학습 후, depth 이미지를 쓰는 student를 imitation으로 학습. Perception-aware 보상은 카메라 yaw와 비행 방향 정렬(예: exp(-|θ_yaw−θ_dir|)).
- **실험:** 시뮬레이션(Columns, Office, Racing 등), HITL로 실제 쿼드로터; 지연 1.4ms, 성공률·비행 시간.
- **주요 기여:** Perception-aware reward로 teacher-student 정책 학습; 최소 시간 + 장애물 회피 + 시야 정렬 통합.
- **본 논문과의 연관성:** “Position만이 아니라 yaw·approach가 중요한 aerial task”의 좋은 예. 본 논문의 Task 2(position+yaw)와 “terminal position만으로는 부족한” 논의를 뒷받침하는 선행 연구로 인용 가능.

---

### 17. Bootstrapping RL with Imitation for Vision-Based Agile Flight (Xing et al., 2024)

- **논문명:** Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight
- **다루는 문제:** 시각 기반 agile flight에서 RL from scratch는 샘플 비효율, IL만 쓰면 전문가 성능 상한·covariate shift 한계.
- **제안 솔루션:** 3단계: (I) privileged state로 teacher RL, (II) teacher→student imitation(시각 입력), (III) student를 시각 기반 RL로 adaptive fine-tune. Catastrophic forgetting 완화를 위해 성능 기반 적응적 업데이트.
- **실험:** 시뮬레이션 및 실제 드론 레이싱; 동일 샘플 예산에서 IL만보다 견고·빠름, RL from scratch가 실패하는 설정에서 성공.
- **주요 기여:** RL+IL 결합으로 시각 기반 agile flight; teacher-student 후 RL fine-tune; 시각만 사용한 레이싱 정책.
- **본 논문과의 연관성:** 시각 기반 agile flight의 최신 방법. 본 논문은 perception/정책 구조가 아니라 critic 초기화이므로, “우리는 정책이 아니라 critic encoder warm-start”로 구분해 인용 가능.

---

### 18. Actor-Critic Model Predictive Control: Differentiable Optimization Meets RL for Agile Flight (Romero et al., 2025, AC-MPC)

- **논문명:** Actor-Critic Model Predictive Control: Differentiable Optimization Meets Reinforcement Learning for Agile Flight (T-RO)
- **다루는 문제:** Model-free RL의 유연성과 MPC의 온라인 재계획·구조를 결합해, agile flight에서 견고성·out-of-distribution 성능·샘플 효율을 모두 얻고 싶음.
- **제안 솔루션:** Actor에 differentiable MPC를 넣어 단기에는 MPC로 제어, 장기에는 critic으로 학습. Cost map(관측→MPC 비용)을 신경망으로 학습. MPVE로 critic 학습 시 MPC 궤적을 활용해 샘플 효율 개선. Value와 MPC 비용 행렬의 관계 분석.
- **실험:** 드론 레이싱 시뮬·실제, 최대 21m/s; AC-MLP 대비 OoD·역학 변화 견고성, L1-MPC 등과 비교.
- **주요 기여:** Actor-critic과 differentiable MPC 통합; MPVE; value–cost 관계 실증 분석; T-RO 게재.
- **본 논문과의 연관성:** Agile flight·RL 맥락의 최신 T-RO 논문. 본 논문은 MPC가 아니라 sparse-reward GC-RL의 critic 초기화이므로, “control 구조가 아닌 critic 표현 초기화”라는 차이를 related work에서 명확히 할 수 있음.

---

## 요약 표 (본 논문과의 연관성 기준)

| 논문 | 그룹 | 본 논문과의 연관성 요약 |
|------|------|--------------------------|
| UVFA | A | Goal-conditioned value/critic의 기초; 본 논문의 f_θ(s,g) 구조의 전제 |
| HER | A | 본 논문이 사용하는 RL backbone; relabeling 유지, critic만 warm-start |
| CHER | A | Replay 선택 curriculum; 본 논문은 critic 초기화만 다룸 |
| HGG | A | Goal 생성; 본 논문은 주어진 goal에 대한 ordering 개선 |
| GCSL | A | Value 제거 vs 본 논문은 value/critic 유지 |
| C-Learning | A | 밀도 관점; 본 논문은 ordering만 사용하는 제한적 해석 |
| CIG | B | Context-conditioned goal vs same-context ordering warm-start |
| LEXA | B | World model continuation vs pairwise ordering만 사용 |
| VGCRL | B | GCRL·MI 통합; 본 논문은 HER 유지 + encoder만 warm-start |
| Contrastive GCRL | B | Generic SSL/contrastive baseline; 본 논문은 same-context ranking으로 구분 |
| RCRL | C | Return contrastive 보조; 본 논문은 ordering 초기화 |
| Lyle et al. | C | 보조 과제·표현 역학; 본 논문은 초기화가 역학에 미치는 영향 논의에 활용 |
| VCR | C | Value-consistent 표현; 목표 유사, 방법 상이(transition vs ordering) |
| Falanga et al. | D | Narrow gap·approach-sensitive aerial; Task 3/4 맥락 |
| Swift | D | 챔피언 수준 드론 레이싱; “critic warm-start”와 차별화 |
| Song et al. | D | Perception-aware·position+yaw; Task 2 논의 지원 |
| Xing et al. | D | Vision-based agile; 정책 vs critic 초기화 구분 |
| AC-MPC | D | T-RO agile flight; 제어 구조 vs critic 초기화 구분 |

---

*이 문서는 `paper/reference_papers/` 내 PDF에서 추출한 텍스트를 바탕으로 작성되었으며, 본 논문(continuation-derived relational initialization for goal-conditioned critics in sparse-reward aerial RL)의 related work 구성 및 초안 작성에 활용할 수 있도록 정리하였다.*

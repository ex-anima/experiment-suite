# Ex Anima Experiment Suite

시연 데이터와 평가 조건을 바꿔 실험을 반복하고, 학습 모델의 행동 재현과 자원 비용을 비교하는 도구다. Core와 Combat Sandbox의 공개 인터페이스나 실행 도구를 사용한다.

## 초기 실험 계획

- 시연 데이터 길이와 기록 주기, 현재 상태와 과거 상태 입력을 비교한다.
- 안정형·적극형 시연을 별도 학습하고 같은 평가 조건에서 비교한다. 학습·평가 데이터는 에피소드 또는 세션 단위로 분리한다.
- 행동 일치율, 클래스별 정밀도·재현율·F1, 혼동 행렬, 전투 행동 분포와 승률을 기록한다.
- 학습 경과 시간, CPU·최대 메모리, 추론 p50/p95 지연, 개체 수별 처리량과 게임 프레임 시간을 측정한다.

데이터·모델·환경·도구의 버전, 설정, 시드와 결과를 함께 기록한다. 자동화 언어, 실행 인터페이스, 결과 형식과 평가 임계값은 상세 설계와 기준선 실험을 거쳐 결정한다.

## Sandbox 사람 시연의 첫 성향별 실험

Python 3.10 이상과 `--policy-stdio`를 지원하는 Sandbox 실행 파일을 사용한다. Ex Anima 작업 공간에서 실행한다. 두 출력 디렉터리에는 새 경로를 지정한다.

```powershell
$pilot = 'results/sandbox-human-pilot/<session>'
python -B experiment-suite/sandbox_decisions.py $pilot datasets/style-pilot --test-seed 111
python -B experiment-suite/sandbox_reference.py datasets/style-pilot combat-sandbox/build/Release/exanima-sandbox.exe results/style-reference
python -B experiment-suite/sandbox_reference.py datasets/style-pilot combat-sandbox/build/Release/exanima-sandbox.exe results/style-reference-range --wait-for-melee-range --rollout-seed 111 --rollout-seed 201 --rollout-seed 202 --rollout-seed 203
```

첫 명령은 사람이 플레이한 양쪽 시연을 10 Hz 단일 범주 판단으로 변환한다. 두 번째 명령은 성향별 최근접 이웃 참조 정책을 만들고, 시연에 대한 오프라인 정확도와 같은 평가 시드에서의 Sandbox 동작을 기록한다. 시연, 변환된 개인 데이터, 참조 정책 파일과 재생 기록은 로컬 경로에 저장된다.

[Sandbox 판단 데이터와 외부 정책 규격](https://github.com/ex-anima/docs/blob/feature/sandbox-first/SANDBOX_DECISIONS.md)

## Core 저장 모델 평가

Core에서 학습 실행 파일과 추론 전용 실행 파일을 각각 빌드한 뒤 Ex Anima 작업 공간에서 실행한다.

```powershell
python -B experiment-suite/core_style.py datasets/style-pilot core/build/full/Release/exanima-train.exe core/build/inference/Release/exanima-infer.exe combat-sandbox/build/Release/exanima-sandbox.exe results/core-style-mlp --balance-classes --rollout-seed 301 --rollout-seed 407 --rollout-seed 509
```

성향별 학습 설정과 모델을 저장한 뒤 추론 전용 CLI를 시작해 시연 판단 일치율을 계산하고 Sandbox를 조작한다. 같은 시드에서 참조 정책도 실행하면 공격 거리, 상대 회복 중 접근·근거리 공격, 피격량과 승패를 비교할 수 있다.

## 저장된 v0.1.0 후보 실행

후보 디렉터리에는 `candidate.json`과 `cautious`, `aggressive` 모델 디렉터리가 있다. 아래 명령은 후보의 모델 해시와 입출력 호환성을 검사하고 두 모델을 동일한 시드로 실행한다. 출력 디렉터리는 새 경로여야 한다.

```powershell
python -B experiment-suite/core_candidate.py models/v0.1.0-candidate core/build/inference/Release/exanima-infer.exe combat-sandbox/build/movement-observation-verified/exanima-sandbox.exe results/candidate-run
```

기본값은 후보에 기록된 시드와 기본 배치다. `--seed N`을 반복 지정해 원하는 시드를 사용할 수 있고, `--layout inset`으로 변경 배치를 실행할 수 있다. 실행 조건은 `protocol.json`, 재생 검증 결과는 `summary.json`, 원본 기록은 `<성향>/<시드>/episode-1.tsv`에 남는다. [외부 정책 규격](https://github.com/ex-anima/docs/blob/9a069aca8ff250a4453f1467799f4277c8e0bdc7/SANDBOX_DECISIONS.md)

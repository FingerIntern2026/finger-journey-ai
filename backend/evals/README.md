# RAG 검색 품질 평가

이 디렉터리는 질문에 맞는 문서 청크가 pgvector 검색 결과 상위에 포함되는지 반복해서 확인하기 위한 평가 도구이다.

## 실행 방법

PostgreSQL을 실행하고 문서 적재를 마친 뒤 `backend` 디렉터리에서 실행한다.

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m evals.evaluate_retrieval --limit 5
Remove-Item Env:HF_HUB_OFFLINE
```

`HF_HUB_OFFLINE=1`은 이미 내려받은 임베딩 모델만 사용하여 불필요한 네트워크 확인을 막는 설정이다.

평가 기준은 다음과 같다.

- `Hit@1`: 정답 문서의 청크가 검색 결과 첫 번째인지 나타내는 비율이다.
- `Hit@5`: 정답 문서의 청크가 상위 5개 안에 있는지 나타내는 비율이다.
- `MRR`: 정답 청크가 앞쪽에 배치될수록 높아지는 순위 품질 지표이다.
- `avg_seconds`: 임베딩 모델 로딩 시간을 제외한 질문당 평균 검색 시간이다.

CI나 수동 검증에서 최소 품질 기준을 적용하려면 다음과 같이 실행한다.

```powershell
.\.venv\Scripts\python.exe -m evals.evaluate_retrieval --limit 5 --fail-under-hit-at-5 0.9
```

## 최초 기준값

2026-09-29에 적재된 문서 35개와 청크 644개를 대상으로 측정한 기준값이다.

| 평가 질문 | Hit@1 | Hit@5 | MRR | 평균 검색 시간 |
| ---: | ---: | ---: | ---: | ---: |
| 15개 | 86.7% | 100% | 0.922 | 0.054초 |

문서나 임베딩 모델, 청킹 방식, 검색 SQL을 변경하면 같은 명령을 다시 실행하여 이 기준값과 비교한다. 실제 사용자 질문에서 실패 사례가 발견되면 `retrieval_cases.json`에 해당 질문과 기대 문서를 추가한다.

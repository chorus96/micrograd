
# micrograd

![awww](puppy.jpg)

작지만 강력한(한 방이 있는! :)) Autograd 엔진입니다. 동적으로 구성된 DAG 위에서 역전파(reverse-mode autodiff)를 구현하고, 그 위에 PyTorch와 유사한 API를 가진 작은 신경망 라이브러리를 얹었습니다. 두 가지 모두 매우 작아서 각각 약 100줄과 50줄 정도의 코드로 이루어져 있습니다. DAG는 스칼라 값에 대해서만 동작하므로, 예를 들어 각 뉴런을 개별적인 아주 작은 덧셈과 곱셈들로 잘게 쪼갭니다. 하지만 이것만으로도 데모 노트북에서 보여주듯이 이진 분류를 수행하는 완전한 심층 신경망을 구성하기에 충분합니다. 교육 목적으로 유용할 수 있습니다.

### DAG란?

DAG는 **Directed Acyclic Graph**(방향성 비순환 그래프)의 약자입니다.

- **Directed (방향성)**: 각 간선(edge)이 방향을 가집니다. 즉 A → B 처럼 한 방향으로만 흐릅니다.
- **Acyclic (비순환)**: 사이클이 없습니다. 어떤 노드에서 출발해 간선을 따라가도 다시 자기 자신으로 돌아오지 않습니다.
- **Graph (그래프)**: 노드(node)와 그 노드들을 잇는 간선(edge)으로 이루어진 자료구조입니다.

micrograd의 맥락에서 DAG는 **연산 그래프(computation graph)**를 의미합니다. 예를 들어 아래 코드가 실행되면 그래프가 동적으로 만들어집니다:

```python
a = Value(-4.0)
b = Value(2.0)
c = a + b
```

```
a ──┐
    ├──(+)──> c
b ──┘
```

- `a`, `b`, `c`가 **노드**이고,
- "`c`는 `a`와 `b`로부터 만들어졌다"는 관계가 **방향성 간선**입니다.
- 계산은 항상 입력에서 출력 방향으로만 흐르고 되돌아오지 않기 때문에 **비순환**입니다.

역전파(backpropagation)는 이 그래프를 **거꾸로** 따라가면서 그래디언트(미분값)를 계산합니다. `g.backward()`를 호출하면, 출력 노드에서 시작해 간선을 역방향으로 거슬러 올라가며 각 노드의 `.grad` 값을 채웁니다. 그래프에 사이클이 없기(acyclic) 때문에 계산 순서가 위상 정렬(topological sort)로 명확하게 정해지며, 이 덕분에 미분 계산이 깔끔하게 이루어집니다.

즉, **DAG는 "어떤 값이 어떤 값으로부터 계산되었는지"의 계보를 기록한 그래프**이고, micrograd는 이 그래프를 자동으로 만들고 거꾸로 훑으면서 자동 미분(autodiff)을 수행하는 엔진입니다.

### 설치

```bash
pip install micrograd
```

### 사용 예시

아래는 지원되는 여러 연산들을 보여주는 다소 작위적인 예시입니다:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # 24.7041 출력, 이 순전파(forward pass)의 결과
g.backward()
print(f'{a.grad:.4f}') # 138.8338 출력, 즉 dg/da의 수치적 값
print(f'{b.grad:.4f}') # 645.5773 출력, 즉 dg/db의 수치적 값
```

### 신경망 학습

`demo.ipynb` 노트북은 2층 신경망(MLP) 이진 분류기를 학습하는 전체 데모를 제공합니다. 이는 `micrograd.nn` 모듈로부터 신경망을 초기화하고, 간단한 SVM "max-margin" 이진 분류 손실을 구현하며, 최적화를 위해 SGD를 사용하여 이루어집니다. 노트북에서 보여주듯이, 두 개의 16노드 은닉층을 가진 2층 신경망을 사용하여 moon 데이터셋에 대해 다음과 같은 결정 경계(decision boundary)를 얻습니다:

![2d neuron](moon_mlp.png)

### 추적 / 시각화

추가적인 편의를 위해, `trace_graph.ipynb` 노트북은 graphviz 시각화를 생성합니다. 예를 들어 아래는 간단한 2D 뉴런의 시각화로, 아래 코드에서 `draw_dot`을 호출하여 얻은 것이며, 각 노드의 데이터(왼쪽 숫자)와 그래디언트(오른쪽 숫자)를 모두 보여줍니다.

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### 테스트 실행

단위 테스트를 실행하려면 [PyTorch](https://pytorch.org/)를 설치해야 합니다. 테스트는 계산된 그래디언트의 정확성을 검증하기 위한 참조로 PyTorch를 사용합니다. 그런 다음 간단히 실행하면 됩니다:

```bash
python -m pytest
```

### 라이선스

MIT

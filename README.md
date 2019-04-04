# Simple paper review
## 1. Introduction
#### Task: Automatically describing the content of an image
__Visual understanding__ 분야는 기존의 Computer vision 문제들, Image classification 혹은 Object recognition 보다 어렵다.  
이미지를 문장으로 설명해내는 Task는 이미지에 포함된 객체와 함께 그것들의 특성과 행동까지 잡아내야 하며, 이를 자연어로 표현해야 하는데, 이는 언어모델까지 필요하다.  
저자들은 이 Task를 풀기 위해 __Machine translation__으로부터 영감을 얻었다.  
문장 내의 단어들을 해석, 치환, 재배치 하는 기존의 방식에서, 최근 __RNN__을 이용한 __Encoder-Decoder__ 방식이 훨씬 간단하고 좋은 성능으로 해결하고 있다.  
__Encoder-RNN__이 문장을 적절한 __Vector representation__으로 변환하고, __Decoder-RNN__이 __Vector__ 로부터 번역한 문장을 생성한다.  
저자들은 이 __Nerual Image Caption__(or __NIC__)라 불리는 __Encoder-RNN__을 __Encoder-CNN__으로 대체한 모델을 제안한다.

Figure 1. 제안한 NIC model의 개요   
![Figure1](resources/figure1.png "figure1")

#### Proposed Model: CNN encoder and RNN decoder(like machine translation), NIC
__CNN__은 기존의 Computer vision 문제들에서 우수한 성능을 내고 있고, 이는 __CNN__이 이미지를 잘 __Embeddeing__ 한다는 것에 설득력이 있다.  
따라서 __CNN__을 이미지 Encoder로 사용하는 것은 자연스러운 일이고, 저자들은 __ImageNet__으로 __Pre-trained__ 된 모델을 사용했다.  
__NIC__는 다른 __NN__과 같이 SGD를 통해 학습된다.  
__NIC__는 __Pascal, Flicker8k, 30k, MSCOCO, SBU__ 데이터셋에서 사람과 근접한 __State of the art__를 달성했다.

## 3. Model
__Machine translation__과 마찬가지로 __Encoder__는 고정된 차원의 __Vector__로 __Embedding__ 하고,  
__Decoder__는 해당 __Vector__를 __Decoding__하여 이미지를 설명하는 문장을 생성한다.  
당시 연구들은 __sequence model__이 주어졌을 때 __correct translation__의 확률을 __maximize__하는 방향으로 학습하는 것이 좋다고 통계적으로 보여진다.  
저자들은 아래와 같은 수식을 __Maximizing__하는 방식으로 모델을 학습시켰다.

Equation 1. Image __*I*__ 가 주어졌을 때, Description __*S*__  
![Eqation1](resources/equation1.png "equation1")

문장 __*S*__ 는 길이가 제한적이지 않고, 따라서 __joint probability__를 이용한 아래 수식처럼 표현할 수 있다.

Equation 2. Joint probability  
![Eqation2](resources/equation2.png "equation2")

문장을 생성하는 모델에 __RNN__을 사용하는것은 자연스럽고, 저자들은 RNN을 더 Concrete하게 만들기 위해 두 가지 Crucial한 선택을 했다.  

#### 1. 어떤 Non-linear function을 써야 학습이 잘 될 것인가.  
이에 대한 문제를 __LSTM__을 사용하여 해결하였다.  
__LSTM__은 __Vanishing or exploding gradient__ 문제를 잘 해결하기 때문에 당시 __Sequence task__에서 __State of the art__를 달성하였고, 저자들도 이를 선택했다.   

#### 2. 어떻게 이미지와 단어를 동시에 입력으로 넣어줄 수 있을까.  
2014 이미지넷에서 우승한 __GoogLeNet__을 사용하여 이미지를 __Embedding__ 했다.  
기존의 Computer vision 문제를 잘 해결하는 모델이 이미지를 잘 표현하는 __Vector representation__를 만들 것이라 생각했다.  
또한 __문장을 Word 단위로 Split__ 하여 __Image vector__와 같은 차원으로 __Embedding__ 했다. 

### 3.1. LSTM based Sentence Generator
__LSTM__은 __Vanishing or exploding gradient__ 문제를 잘 해결하기 때문에 선택했고, __LSTM__의 전체적인 구조는 아래와 같다.  

Figure 2. LSTM structure  
![Figure2](resources/figure2.png "figure2")

__Encoder-CNN__과 결합한 LSTM의 모습은 아래와 같고, 모든 LSTM의 parameter는 공유된다.

Figure 3. LSTM model combined with a CNN image embedder  
![Figure3](resources/figure3.png "figure3")

이미지는 맨 처음 입력 단 한번만 들어가고, 이미지 벡터로부터 LSTM이 출력한 결과를 다음 LSTM의 입력으로 넣으면서 학습, 추론한다.  
저자들은 매 step마다 이미지를 넣어주는 시도를 했으나 이는 오히려 더 쉽게 Overfit 되는 결과를 보였다.  

#### Inference
저자들은 __NIC__에서 __Inference__하는 두 가지 방법을 제시했다.  

##### Sampling
__Sampling__ 방식은 아주 단순하다.  
최대 문장의 길이가 될 때 혹은 끝나는 신호가 나올 때까지 __LSTM__에서 __Softmax__를 거쳐 출력된 최대 값의 단어를 이어붙여, 이를 다음 LSTM의 입력으로 넣어주는 방식이다.

##### BeamSearch
__Beamsearch__는 매 t번째까지의 입력으로 만들어진 문장 k개를 유지하며, k개의 문장들로부터 t+1번째까지의 문장 중 다시 k개를 반환하는 방식이다.  
k=1 일 때는 굉장히 Greedy하며 BLEU score는 k=20일 때의 BLEU score보다 평균적으로 2점 정도 하락했다.  
 
## 4. Experiments
### 4.1 Evaluation Metrics
__Image Description__에서 가장 많이 사용되는 __Metric__은 __BLEU score__이고, __n-gram__을 통해 평가된다.  
__n-gram__이란 다음에 나올 단어를 예측할 때 은 앞선 __n-1__개의 단어에 의존하는 방식이다.  
주로 __1-gram__을 많이 이용하고 저자들 또한 __BLEU-1__을 주로 이용하였고, 추가적으로 ___ME-TEOR, Cider score___도 제시하였다.

### 4.2. Datasets

다음과 같은 Dataset을 이용하였다.  

Table 1. Datasets.  
![table1](resources/table1.png "table1")

__SBU__를 제외하고는 __모두 5개의 문장__이 __Labeling__ 되어 있다.  
저자들은 __SBU__가 __Flikcr__에서 사용자들이 올린 Description이기 때문에 Noise가 있다고 보았다.  
또한 __Pascal__은 Test를 위해서만 사용하였는데, 나머지 4개의 Data로 학습을 하고 Testing 하였다.  

### 4.3. Result
#### 4.3.1. Training Details

논문에서 저자들은 아래와 같은 사항들을 이용하여 실험했다.

##### 1. 학습 시 __CNN__은 __Imagnet__을 통해 __Pre-trained__ 된 __Weight__를 그대로 이용하였고, __Fine tuning__은 하지 않았다.  
##### 2. __Word embedding vector__도 __pre-trained__ 된 모델을 써 보았으나 효과가 없었다.  
##### 3. __SGD__로 학습했고 __fixed learning rate__를 사용, __decay__는 사용하지 않았다.  
##### 4. __Word embedding size__와 LSTM의 크기는 512로 셋팅했다.  
##### 5. __Overfitting__을 피하기 위해 __Dropout__과 __ensemble__을 사용했다. - BLEU score 향상은 거의 없었다.  
##### 6. __Hidden layer__의 개수와 깊이를 다양하게 설정했다.  

#### 4.3.2. Generation Results

Table 2. BLEU-1 score.  
![table2](resources/table2.png "table2")

사용한 4가지 Datasets 전부에서 __SOTA BLEU-1 score__를 갱신했다.  

#### 4.3.3. Transfer Learning, Data Size and Label Quality

어떤 __datasets__로부터 학습된 모델은 다른 d__atasets__에도 적용될 수 있는지 실험했다.  
같은 유저 집단이 만든 __Flickr dataset__에서는 __Transfer learning__이 효과가 있었다.  
__Flickr30k__로 학습된 모델을 이용하면 __Flickr8k__에서 __BLEU score__가 4점 정도 상승한다.  
__MSCOCO__는 __Flickr30k__ 보다 5배 크지만, __dataset 간 __miss-match__가 많았고 __BLEU score__가 10점 정도 내려갔다.  

#### 4.3.4. Generation Diversity Discussion

저자들은 __Generate model__ 관점에서, 얼마나 다양한 문장들을 생성할 수 있는지, 얼마나 수준이 높은지를 확인하였다.  
__k=20__인 __BeamSearch__에서 상위 15개 정도는 __BLEU-1 score__ 기준으로 사람과 견줄만한 __58__점 정도를 달성했다.  

Table 3. MSCOCO test set의 생성된 몇가지 예시.  
![table3](resources/table3.png "table3")

#### 4.3.7. Analysis of Embeddings
__one-hot encoding__과 다르게 __Embedding__은 Word dictionary의 크기에 제한되지 않는다.  
저자들은 __Embedding space__에서 __KNN__을 이용한 몇 가지 예시를 제시했는데 아래와 같다.  

Table 6. KNN을 이용한 Word embedding space analysis.  
![table6](resources/table6.png "table6")

__Embedding Vector__는 다른 __Vision component__에도 도움을 줄 수 있는데,  
__horse__와 __pony, donkey__는 근접하기 때문에 __CNN__이 __horse-looking__ 동물의 특징을 추출하는 것이 더 수월해질 것이다.  
아주 심한 경우에 __Unicorn__ 같은 몇몇 예시들도 __horse__와 근접하기 때문에, 기존의 __bag of word__ 방식보다 더 많은 정보를 줄 수 있다.

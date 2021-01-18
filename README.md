# Show and Tell: A Neural Image Caption Generator

I implemented the code using Keras.  
Requirements: Python3, Keras 2.0(Tensorflow backend), NLTK, matplotlib, PIL, h5py, Jupyter  

# Training and testing

1. Download the Flicker8k dataset and place it in the path that contains the notebook file.  
Maybe the directory names are Flicker8k_Dataset and Flickr8k_text.  
Flicker8k_Dataset contains image files, and Flickr8k_text contains caption and dataset split information.(train, dev, test)  

2. Data prepare
Open data_processing.ipynb  
In most cases, the code have comments and can be followed in order.  

3. Train and test
Open train_evaluate.ipynb  
In most cases, the code have comments and can be followed in order.  

# Demo

1. Open Demo.ipynb  
In demo, custom data to be put, and we have prepared an example in ./examples/.  
2. If you want to only demo, final model is in the repository, so you just need to create a tokenizer in train_evaluate.ipynb.  

# Simple paper review
## 1. Introduction
#### Task: Automatically describing the content of an image
**Visual understanding** 분야는 기존의 Computer vision 문제들, Image classification 혹은 Object recognition 보다 어렵다.  
이미지를 문장으로 설명해내는 Task는 이미지에 포함된 객체와 함께 특성과 행동까지 잡아내야 하며, 이를 자연어로 표현해야 하는데 이는 이미지 모델뿐 아니라 언어모델까지 필요하다.  
저자들은 이 Task를 풀기 위해 **Machine translation**으로부터 영감을 얻었다.  
문장 내의 단어들을 해석, 치환, 재배치 하는 기존의 방식에서, 최근 **RNN**을 이용한 **Encoder-Decoder** 방식이 훨씬 간단하고 좋은 성능으로 해결하고 있다.  
**Encoder-RNN**이 문장을 적절한 **Vector representation**으로 변환하고, **Decoder-RNN**이 **vector** 로부터 번역한 문장을 생성한다.  
저자들은 이 **Nerual Image Caption**(or **NIC**)라 불리는 **Encoder-RNN**을 **Encoder-CNN**으로 대체한 모델을 제안한다.

Figure 1. 제안한 NIC model의 개요   
![figure1](https://user-images.githubusercontent.com/15686143/55540612-4cb31f00-56fe-11e9-913d-953e02738736.PNG)

#### Proposed Model: CNN encoder and RNN decoder(like machine translation), NIC
**CNN**은 기존의 Computer vision 문제들에서 우수한 성능을 내고 있고, 이는 **CNN**이 이미지를 잘 **embeddeing** 한다고 볼 수 있다.  
따라서 **CNN**을 이미지 encoder로 사용하는 것은 자연스러운 일이고, 저자들은 **ImageNet**으로 **pre-trained** 된 모델을 사용했다.  
**NIC**는 다른 **NN**과 같이 SGD를 통해 학습된다.  
**NIC**는 **Pascal, Flicker8k, 30k, MSCOCO, SBU dataset**에서 사람과 근접한 **State of the art**를 달성했다.

## 3. Model
**Machine translation**과 마찬가지로 **Encoder**는 고정된 차원의 **vector**로 **encoding** 하고,  
**Decoder**는 해당 **vector**를 **decoding**하여 이미지를 설명하는 문장을 생성한다.  
당시 연구들은 **sequence model**이 주어졌을 때 **correct translation**의 확률을 **maximize**하는 방향으로 학습하는 것이 좋다고 통계적으로 보여진다.  
저자들은 아래와 같은 수식을 **Maximizing**하는 방식으로 모델을 학습시켰다.

Equation 1. Image ***I*** 가 주어졌을 때, Description ***S***  
![equation1](https://user-images.githubusercontent.com/15686143/55540609-4cb31f00-56fe-11e9-9734-ab1e665651b7.PNG)

문장 ***S*** 는 길이가 제한적이지 않고, 따라서 **joint probability**를 이용한 아래 수식처럼 표현할 수 있다.

Equation 2. Joint probability  
![equation2](https://user-images.githubusercontent.com/15686143/55540611-4cb31f00-56fe-11e9-86e7-e7b090f5a865.PNG)

문장을 생성하는 모델에 **RNN**을 사용하는것은 자연스럽고, 저자들은 RNN을 더 Concrete하게 만들기 위해 두 가지 Crucial한 선택을 했다.  

#### 1. 어떤 Non-linear function을 써야 학습이 잘 될 것인가.  
이에 대한 문제를 **LSTM**을 사용하여 해결하였다.  
**LSTM**은 **Vanishing or exploding gradient** 문제를 잘 해결하기 때문에 당시 **sequence task**에서 **State of the art**를 달성하였고, 저자들도 이를 선택했다.   

#### 2. 어떻게 이미지와 단어를 동시에 입력으로 넣어줄 수 있을까.  
2014 이미지넷에서 우승한 **GoogLeNet**을 사용하여 이미지를 **embedding** 했다.  
기존의 Computer vision 문제를 잘 해결하는 모델이 이미지를 잘 표현하는 **vector representation**를 만들 것이라 생각했다.  
또한 **문장을 word 단위로 split** 하여 **Image vector**와 같은 차원으로 **embedding** 했다. 

### 3.1. LSTM based Sentence Generator
**LSTM**은 **Vanishing or exploding gradient** 문제를 잘 해결하기 때문에 선택했고, **LSTM**의 전체적인 구조는 아래와 같다.  

Figure 2. LSTM structure  
![figure2](https://user-images.githubusercontent.com/15686143/55540614-4cb31f00-56fe-11e9-8fc5-baaa663d82fe.PNG)

**Encoder-CNN**과 결합한 LSTM의 모습은 아래와 같고, 모든 LSTM의 parameter는 공유된다.

Figure 3. LSTM model combined with a CNN image embedder  
![figure3](https://user-images.githubusercontent.com/15686143/55540615-4d4bb580-56fe-11e9-80c5-53b0fe531253.PNG)

이미지는 맨 처음 입력 단 한번만 들어가고, 이미지 벡터로부터 LSTM이 출력한 결과를 다음 LSTM의 입력으로 넣으면서 학습, 추론한다.  
저자들은 매 step마다 이미지를 넣어주는 시도를 했으나 이는 오히려 더 쉽게 Overfit 되는 결과를 보였다.  

#### Inference
저자들은 **NIC**에서 **Inference**하는 두 가지 방법을 제시했다.  

##### Sampling
**Sampling** 방식은 아주 단순하다.  
최대 문장의 길이가 될 때 혹은 끝나는 신호가 나올 때까지 **LSTM**에서 **Softmax**를 거쳐 출력된 최대 값의 단어를 이어붙여, 이를 다음 LSTM의 입력으로 넣어주는 방식이다.

##### BeamSearch
**BeamSearch**는 매 t번째까지의 입력으로 만들어진 문장 k개를 유지하며, k개의 문장들로부터 t+1번째까지의 문장 중 다시 k개를 반환하는 방식이다.  
k=1 일 때는 굉장히 Greedy하며 BLEU score는 k=20일 때의 BLEU score보다 평균적으로 2점 정도 하락했다.  
 
## 4. Experiments
### 4.1 Evaluation Metrics
**Image description**에서 가장 많이 사용되는 **Metric**은 **BLEU score**이고, **n-gram**을 통해 평가된다.  
**n-gram**이란 다음에 나올 단어를 예측할 때 은 앞선 **n-1**개의 단어에 의존하는 방식이다.  
주로 **1-gram**을 많이 이용하고 저자들 또한 **BLEU-1**을 주로 이용하였고, 추가적으로 **ME-TEOR, Cider score**도 제시하였다.

### 4.2. Datasets

다음과 같은 Dataset을 이용하였다.  

Table 1. Datasets.  
![table1](https://user-images.githubusercontent.com/15686143/55540616-4d4bb580-56fe-11e9-9c5f-1b18fa4878c7.PNG)

**SBU**를 제외하고는 **모두 5개의 문장**이 **Labeling** 되어 있다.  
저자들은 **SBU**가 **Flikcr**에서 사용자들이 올린 Description이기 때문에 Noise가 있다고 보았다.  
또한 **Pascal**은 Test를 위해서만 사용하였고, 나머지 4개의 Data로 학습한 모델을 이용했다.  

### 4.3. Result
#### 4.3.1. Training Details

논문에서 저자들은 아래와 같은 사항들을 이용하여 실험했다.

1. 학습 시 **CNN**은 **Imagnet**을 통해 **pre-trained** 된 **weight**를 그대로 이용하였고, **fine tuning**은 하지 않았다.  
2. **Word embedding vector**도 **pre-trained** 된 모델을 써 보았으나 효과가 없었다.  
3. **SGD**로 학습했고 **fixed learning rate**를 사용, **decay**는 사용하지 않았다.  
4. **Word embedding size**와 LSTM의 크기는 512로 셋팅했다.  
5. **Overfitting**을 피하기 위해 **Dropout**과 **ensemble**을 사용했다. - BLEU score 향상은 거의 없었다.  
6. **Hidden layer**의 개수와 깊이를 다양하게 설정했다.  

#### 4.3.2. Generation Results

Table 2. BLEU-1 score.  
![table2](https://user-images.githubusercontent.com/15686143/55540617-4d4bb580-56fe-11e9-939d-029437bb2094.PNG)

사용한 4가지 Datasets 전부에서 **SOTA BLEU-1 score**를 갱신했다.  

#### 4.3.3. Transfer Learning, Data Size and Label Quality

어떤 **datasets**로부터 학습된 모델은 다른 **datasets**에도 적용될 수 있는지 실험했다.  
같은 유저 집단이 만든 **Flickr dataset**에서는 **Transfer learning**이 효과가 있었다.  
**Flickr30k**로 학습된 모델을 이용하면 **Flickr8k**에서 **BLEU score**가 4점 정도 상승한다.  
**MSCOCO**는 **Flickr30k** 보다 5배 크지만, **dataset 간 **miss-match**가 많았고 **BLEU score**가 10점 정도 내려갔다.  

#### 4.3.4. Generation Diversity Discussion

저자들은 **Generate model** 관점에서, 얼마나 다양한 문장들을 생성할 수 있는지, 얼마나 수준이 높은지를 확인하였다.  
**k=20**인 **BeamSearch**에서 상위 15개 정도는 **BLEU-1 score** 기준으로 사람과 견줄만한 **58**점 정도를 달성했다.  

Table 3. MSCOCO test set의 생성된 몇가지 예시.  
![table3](https://user-images.githubusercontent.com/15686143/55540618-4de44c00-56fe-11e9-9865-3fff3e9e4bf1.PNG)

#### 4.3.7. Analysis of Embeddings
**one-hot encoding**과 다르게 **Embedding**은 **Word dictionary**의 크기에 제한되지 않는다.  
저자들은 **Embedding space**에서 **KNN**을 이용한 몇 가지 예시를 제시했는데 아래와 같다.  

Table 6. KNN을 이용한 Word embedding space analysis.  
![table6](https://user-images.githubusercontent.com/15686143/55540621-4de44c00-56fe-11e9-880a-fbcfbbbd3663.PNG)

**Embedding Vector**는 다른 **Vision component**에도 도움을 줄 수 있는데,  
**horse**와 **pony, donkey**는 근접하기 때문에 **CNN**이 **horse-looking** 동물의 특징을 추출하는 것이 더 수월해질 것이다.  
아주 심한 경우에 **Unicorn** 같은 몇몇 예시들도 **horse**와 근접하기 때문에, 기존의 **bag of word** 방식보다 더 많은 정보를 줄 수 있다.

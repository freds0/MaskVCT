## Análise do MaskVCT (arXiv:2509.17143)

### O que é

O MaskVCT é um modelo de voice conversion (VC) zero-shot que oferece controlabilidade multi-fator por meio de múltiplos classifier-free guidances (CFGs). Enquanto modelos anteriores de VC se baseiam em um esquema de condicionamento fixo, o MaskVCT integra diversas condições em um único modelo.

O trabalho é de pesquisadores do Center for Language and Speech Processing da Johns Hopkins University.

---

### Contribuições principais

**1. Representação linguística silábica (SylBoost)**

O modelo utiliza representações silábicas do SylBoost para ganhar flexibilidade na modulação de pitch, seleção de fonemas e variação de sotaque dentro de cada segmento. A ideia central é que features SSL convencionais a 25–50 Hz "vazam" pitch e timbre do locutor — o SylBoost opera a 8.33 Hz, sendo temporalmente mais grosseiro e naturalmente mais despido dessas informações parasitas.

**2. Dual path linguístico**

O modelo suporta dois modos de condicionamento linguístico: tokens silábicos discretos (melhor similaridade de locutor) ou features contínuas via projeção FFN (melhor inteligibilidade). Treina com 50% de cada, deixando o usuário escolher no momento da inferência.

**3. Triple Classifier-Free Guidance**

Na VC, onde as saídas devem satisfazer simultaneamente múltiplos fatores de condicionamento — incluindo identidade do locutor, conteúdo linguístico e contorno de pitch — os autores estendem o CFG para três guidances combinados. Os pesos ω_all, ω_spk e ω_ling são ajustáveis na inferência.

**4. Dois modos de inferência**

A partir da mesma arquitetura treinada, propõem:
- `MaskVCT-All` — condicionado em pitch + locutor + linguística; maior fidelidade prosódica.
- `MaskVCT-Spk` — condicionado em locutor + linguística (sem pitch); maior similaridade de locutor.

**5. Codec + mascaramento**

O modelo emprega o checkpoint oficial do DAC 16 kHz como codec, extraindo 9 índices de codebook para codificar cada frame de áudio. Para melhorar a robustez a inconsistências de fase, aplicam PhaseAug a todas as entradas antes do encoder DAC.

---

### Arquitetura técnica

Utiliza uma arquitetura simples baseada em Transformer encoder com PreLN e rotary positional embedding (RoPE), com 16 camadas, 16 attention heads, dimensão de modelo 1024 e dimensão de FFN 4096. Após as camadas, há classification heads separadas para cada índice de codebook.

---

### Resultados

O MaskVCT-Spk alcança a maior similaridade de locutor alvo e a maior similaridade de sotaque em comparação com os baselines. O modelo tem 234M parâmetros (mais 94M + 74M de pré-treinados) e foi treinado em 100K horas de dados, em 2 GPUs A100 por 250k steps.

---

### Repositórios base para implementar o MaskVCT

O artigo não disponibiliza código próprio (até a data do paper), mas todos os seus componentes principais têm código aberto:

---

**🔵 MaskGCT — modelo inspirador mais direto**

O MaskGCT tem código e checkpoints disponíveis em `https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct`. Este é o repositório mais relevante: o MaskGCT é um modelo TTS totalmente não-autoregressivo que elimina a necessidade de alinhamento explícito entre texto e fala. O MaskVCT reusa exatamente o paradigma de mascaramento do módulo S2A do MaskGCT — inclusive a estratégia de codebook sampling, o schedule de mascaramento cossenoidal e a loss de classificação. O repo Amphion contém também o baseline `MaskGCT-S2A` testado no próprio paper.

> **GitHub:** [`open-mmlab/Amphion`](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)

---

**🟣 SyllableLM/SylBoost — extrator linguístico silábico**

O código e checkpoints do SyllableLM estão disponíveis em `https://www.github.com/alanbaade/SyllableLM`. O repositório inclui exemplos de uso do `SylBoostFeatureReader` com suporte às taxas de 8.33 Hz, 6.25 Hz e 5.0 Hz, e não requer Fairseq — o Data2Vec2 foi copiado e simplificado.

> **GitHub:** [`AlanBaade/SyllableLM`](https://github.com/AlanBaade/SyllableLM)

---

**🟢 DAC — codec de áudio**

O Descript Audio Codec (DAC) está disponível em [`descriptinc/descript-audio-codec`](https://github.com/descriptinc/descript-audio-codec) e também no HuggingFace. O MaskVCT usa o checkpoint oficial 16 kHz com 9 codebooks.

---

**🔴 Baselines com código aberto mencionados no paper**

| Modelo | Repositório |
|--------|-------------|
| Diff-HierVC | [`hayeong0/Diff-HierVC`](https://github.com/hayeong0/Diff-HierVC) |
| FreeVC | [`OlaWod/FreeVC`](https://github.com/OlaWod/FreeVC) |
| GenVC | [`caizexin/GenVC`](https://github.com/caizexin/GenVC) |
| FACodec | [HuggingFace Amphion](https://hf.co/amphion/naturalspeech3_facodec) |

---

### Estratégia recomendada para implementar o MaskVCT

A rota mais direta é partir do **Amphion/MaskGCT** como base, pois o núcleo do modelo (Transformer mascarado, codebook sampling, CFG) já está implementado. As adaptações necessárias em cima dele são:

1. Substituir o condicionamento textual pelo condicionamento de fala fonte (tokens DAC + SylBoost + Praat pitch)
2. Adicionar o mecanismo de speaker prompt via concatenação (estilo VALL-E)
3. Estender o CFG de dual para triple, conforme a Equação 3 do paper
4. Integrar o `SylBoostFeatureReader` do repositório SyllableLM como extrator linguístico
5. Adicionar PhaseAug na entrada do DAC encoder
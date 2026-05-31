# Detecção de Trincas e Fissuras em Paredes

**Sistema de inspeção automatizada com segmentação de instância** — Desafio 2 do Programa de Residência em Inteligência Artificial (UNISENAI 2026).

Autor: **Daniel Tavares de França**

## Sobre o projeto

Solução de visão computacional para detectar e segmentar trincas e fissuras em paredes a partir de imagens. O objetivo é apoiar a inspeção de qualidade na construção civil, hoje feita manualmente e sujeita a fadiga, reduzindo retrabalho nas etapas de revestimento e pintura.

O modelo foi treinado para ser **leve**, viabilizando execução em dispositivos de baixa capacidade de processamento.

## Demonstração ao vivo

Aplicação web pública (Gradio + Hugging Face Spaces):

➡️ **https://huggingface.co/spaces/DanielTF/detector-trincas**

## Abordagem

- **Modelo:** YOLOv8n-seg (segmentação de instância, versão *nano*), com transferência de aprendizado a partir do COCO.
- **Tamanho:** 6,8 MB · 3.258.259 parâmetros · 11,3 GFLOPs.
- **Dados:** 1.551 imagens anotadas com polígonos de segmentação (classe única: `trinca`), divididas em 80% treino / 20% validação (`seed=42`).
- **Treinamento:** 50 épocas · lote 16 · 640×640 · GPU Tesla T4 · ~49 minutos.

## Resultados (conjunto de validação: 311 imagens, 423 instâncias)

| Métrica      | Máscara (segmentação) | Caixa (detecção) |
|--------------|:---------------------:|:----------------:|
| Precisão     | 76,4%                 | 80,0%            |
| Recall       | 59,1%                 | 66,4%            |
| mAP@50       | 62,3%                 | 73,4%            |
| mAP@50–95    | 22,6%                 | 55,2%            |

Tempo de inferência ~5,5 ms/imagem na GPU Tesla T4.

## Estrutura do repositório

```
.
├── Desafio_2.ipynb       # Pipeline completo: dados, treino, avaliação e exportação
├── app.py                # Aplicação web (Gradio)
├── requirements.txt      # Dependências
├── Relatorio_Tecnico.pdf # Relatório técnico do projeto
└── README.md
```

## Como executar

### Aplicação web (local)
```bash
pip install -r requirements.txt
python app.py
```

### Notebook
O notebook `Desafio_2.ipynb` foi desenvolvido para o Google Colab (com GPU) e reproduz todo o processo de preparação dos dados, treinamento, avaliação e exportação do modelo.

## Observações

- O **dataset** e os **pesos do modelo** não são versionados neste repositório por questão de tamanho. O dataset foi fornecido pela organização do desafio.
- O relatório técnico detalha o problema, a metodologia, as métricas e as limitações da solução.

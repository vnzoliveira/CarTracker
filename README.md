# 🚗 Detector de Veículos com Estimativa de Velocidade

Este projeto utiliza **YOLOv8** para detecção de veículos em vídeos e estima sua velocidade com base no tempo entre duas linhas horizontais. Também recorta automaticamente imagens da região da placa dos veículos e as salva para uso posterior (como OCR).

---

## 📁 Estrutura do Projeto

```
CPO2-DATAVIZ/
│
├── assets/
│   ├── placas/                # Recortes das placas serão salvos aqui
│   └── example.mp4           # Vídeo de entrada para análise
│
├── dependencies/
│   └── requirements.txt      # Dependências do projeto
│
├── main/
│   └── main.py               # Código principal do projeto
```

---

## ✅ Requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)
- Ambiente virtual (opcional, mas recomendado)

---

## 📦 Instalação

1. **Clone o repositório (ou baixe o ZIP):**
   ```bash
   git clone https://github.com/seu-usuario/CPO2-DATAVIZ.git
   cd CPO2-DATAVIZ
   ```

2. **Crie e ative um ambiente virtual:**

   - No Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

   - No Linux/macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Instale as dependências:**
   ```bash
   pip install -r dependencies/requirements.txt
   ```

---

## 📥 Dependências

O projeto depende dos seguintes pacotes:

- `ultralytics` – Para carregar e usar modelos YOLOv8
- `opencv-python` – Processamento de vídeo
- `numpy` – Cálculos e manipulações numéricas

Instalação manual (caso não use o `requirements.txt`):
```bash
pip install ultralytics opencv-python numpy
```

---

## ▶️ Como Rodar

1. Coloque o vídeo desejado dentro da pasta `assets`.

2. Execute o script com o comando:

```bash
python main/main.py --video assets/example.mp4 --modelo yolov8n.pt
```

### Argumentos disponíveis:

| Argumento       | Descrição                                               | Valor padrão         |
|------------------|-----------------------------------------------------------|----------------------|
| `--video`        | Caminho para o vídeo de entrada                           | `assets/example.mp4` |
| `--modelo`       | Caminho do modelo YOLOv8 (`.pt`)                          | `yolov8n.pt`         |
| `--distancia`    | Distância real (em metros) entre as duas linhas          | `75.0`               |
| `--linha1`       | Posição Y da primeira linha (pixels)                     | `150`                |
| `--linha2`       | Posição Y da segunda linha (pixels)                      | `700`                |
| `--conf`         | Confiança mínima da detecção                             | `0.3`                |
| `--escala`       | Redução da escala do vídeo (para melhorar desempenho)    | `0.5`                |
| `--skip`         | Número de frames a pular entre processamentos            | `2`                  |

3. Durante a execução, uma janela será aberta exibindo as detecções em tempo real.

> Pressione **ESC** para encerrar a visualização.

---

## 🧠 Funcionalidades

- ✔️ Detecção de veículos usando YOLOv8
- ✔️ Estimativa de velocidade com base em duas linhas horizontais
- ✔️ Rastreamento simples de objetos via centroides
- ✔️ Recorte automático das regiões de placa e salvamento em `assets/placas`
- ✔️ Exibição de FPS, velocidade estimada e contagem de veículos ativos

---

## 🛠️ Melhorias Futuras

- [ ] Integrar OCR (EasyOCR ou Tesseract) para leitura das placas
- [ ] Exportar dados para CSV
- [ ] Interface web simples com Streamlit ou Flask
- [ ] Suporte a múltiplos vídeos simultaneamente

---

## 🖼️ Exemplo de Execução

```bash
python main/main.py --video assets/example.mp4 --modelo yolov8n.pt --linha1 120 --linha2 600
```


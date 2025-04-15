import cv2
import time
import numpy as np
import os
from ultralytics import YOLO
import argparse

#argumentos a serem passados para manipular parametros de calculo e analise de frames.
def parse_args():
    parser = argparse.ArgumentParser(description="Detector de veículos com estimativa de velocidade")
    parser.add_argument("--video", default="../assets/example.mp4", help="Caminho para o vídeo")
    parser.add_argument("--modelo", default="yolov8n.pt", help="Modelo YOLO (yolov8n.pt recomendado)")
    parser.add_argument("--distancia", type=float, default=75.0, help="Distância real entre linhas (metros)")
    parser.add_argument("--linha1", type=int, default=150, help="Posição Y da primeira linha")
    parser.add_argument("--linha2", type=int, default=700, help="Posição Y da segunda linha")
    parser.add_argument("--conf", type=float, default=0.3, help="Confiança mínima de detecção")
    parser.add_argument("--escala", type=float, default=0.5, help="Fator de escala do frame para processamento")
    parser.add_argument("--skip", type=int, default=2, help="Processar a cada N frames")
    return parser.parse_args()

#Melhora imagens para destacamento das letras, usando CLAHE
def melhorar_imagem_placa(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #aumenta contraste, sem estouro de area. 
    equ = clahe.apply(gray) 
    gaussian = cv2.GaussianBlur(equ, (0, 0), 3.0) #Desfoque Gaussiano, suavizar ruídos.
    nitido = cv2.addWeighted(equ, 1.5, gaussian, -0.5, 0) #Imagem_nítida = Original * α + Suavizada * β
    resultado = cv2.cvtColor(nitido, cv2.COLOR_GRAY2BGR)
    return resultado

def main():
    args = parse_args()
    device = 'cpu'
    PASTA_PLACAS = r"..\cp02-dataviz\assets\placas"
    os.makedirs(PASTA_PLACAS, exist_ok=True)

    print(f"[INFO] Carregando modelo {args.modelo}...")
    try:
        modelo = YOLO(args.modelo)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelo: {e}")
        print("[DICA] Se estiver usando YOLOv5, considere mudar para YOLOv8 para melhor desempenho")
        return

    #Params alvo do YOLO. (alterar conforme vídeo.)
    classes_alvo = ['car', 'truck', 'bus', 'motorcycle', 'van']

    
    print(f"[INFO] Abrindo vídeo: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) #Frames do video.
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolução: {largura}x{altura}, FPS: {fps}")

    #tracking anti oclusao.
    rastreamento = {} #dividir carros em id dispersos e fazer o rastreamento por ID.
    id_carro = 0
    max_frames_ausente = 30

    frame_count = 0
    ultimo_fps = time.time()
    fps_real = 0

    print(f"[INFO] Iniciando processamento. Pressione ESC para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #verifica se o frame é processado ou é pulado.
        frame_count += 1
        tempo_atual = time.time()
        frame_original = None
        eh_frame_processado = frame_count % args.skip == 0

        #Linhas de ref.
        cv2.line(frame, (0, args.linha1), (largura, args.linha1), (255, 255, 0), 2)
        cv2.line(frame, (0, args.linha2), (largura, args.linha2), (255, 255, 0), 2)
        cv2.putText(frame, "Linha 1", (10, args.linha1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Linha 2", (10, args.linha2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        #Otimiza o processamento do modelo YOLO, rodando ele apenas a cada N frame definido.
        if eh_frame_processado:
            frame_original = frame.copy()
            if args.escala != 1.0:
                frame_proc = cv2.resize(frame, (int(largura * args.escala), int(altura * args.escala))) #resize na escala normal do frame (largura x altura normalizada.)
            else:
                frame_proc = frame.copy()

            resultados = modelo(frame_proc, conf=args.conf, device=device, verbose=False)[0] #YOLO pega o frame e processa.

            #Tracking manual por centro do objeto
            for cid in rastreamento:
                rastreamento[cid]['detectado'] = False

            #deteccoes feitas pelo YOLO
            for det in resultados.boxes: 
                cls_id = int(det.cls.item())
                label = modelo.names[cls_id]

                #Ignora objetos fora das classes alvo.
                if label not in classes_alvo:
                    continue
                
                #Se o video foi redimensionado, converte as coordenadas para escala real do vídeo.
                if args.escala != 1.0:
                    coords = det.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords / args.escala)
                else:
                    x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                carro_detectado = False

                #percorre os dados de rastreamento por id e verifica se passou da linha para calcular velocidade em detrimento da distancia e tempo
                for cid, dados in rastreamento.items():
                    largura_bbox = x2 - x1
                    altura_bbox = y2 - y1
                    #Calculo essencial da distancia usando distancia euclidiana (distancia entre 2 pontos no espaço.)
                    dist_max = max(50, min(largura_bbox, altura_bbox) * 0.4) 
                    dist = np.sqrt((dados['cx'] - cx)**2 + (dados['cy'] - cy)**2)

                    if dist < dist_max:
                        rastreamento[cid]['cx'] = cx
                        rastreamento[cid]['cy'] = cy
                        rastreamento[cid]['box'] = (x1, y1, x2, y2)
                        rastreamento[cid]['detectado'] = True
                        rastreamento[cid]['frames_ausente'] = 0
                        rastreamento[cid]['ativo'] = True
                        rastreamento[cid]['label'] = label
                        carro_detectado = True

                        if 'linha1' not in dados and abs(cy - args.linha1) < 10:
                            rastreamento[cid]['linha1'] = tempo_atual
                            #print(f"[INFO] Veículo {cid} ({label}) passou pela linha 1")

                        if 'linha2' not in dados and abs(cy - args.linha2) < 10:
                            rastreamento[cid]['linha2'] = tempo_atual

                            if 'linha1' in dados:
                                tempo_decorrido = rastreamento[cid]['linha2'] - rastreamento[cid]['linha1']

                                if tempo_decorrido > 0:
                                    distancia_ajustada = args.distancia
                                    if args.skip > 1:
                                        distancia_ajustada = distancia_ajustada / args.skip

                                    velocidade = (distancia_ajustada / tempo_decorrido) * 3.6
                                    rastreamento[cid]['velocidade'] = velocidade
                                    rastreamento[cid]['tempo_decorrido'] = tempo_decorrido
                                    #print(f"[INFO] Veículo {cid} ({label}) → {velocidade:.1f} km/h (tempo: {tempo_decorrido:.2f}s)")

                                    #Se o frame rastreado corresponde a um carro identificado e seu ID, recorta a placa.
                                    if frame_original is not None and not rastreamento[cid].get('placa_capturada', False):
                                        #Calcula recorte pela altura relativa do bbox do carro.
                                        altura_bbox = y2 - y1
                                        recorte_y1 = max(0, y2 - int(0.25 * altura_bbox))
                                        margem = int(0.1 * (x2 - x1))
                                        x1_placa = max(0, x1 - margem)
                                        x2_placa = min(largura, x2 + margem)
                                        recorte = frame_original[recorte_y1:y2, x1_placa:x2_placa]

                                        #se o tamanho do recorte é maior que 0 (existe), trata imagem, salva na pasta e flaga que foi capturada para aquele ID em rastreamento.
                                        if recorte.size > 0:
                                            recorte_zoom = cv2.resize(recorte, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                                            recorte_melhorado = melhorar_imagem_placa(recorte_zoom)

                                            nome_arquivo = f"veiculo_{cid}_{label}_{velocidade:.1f}kmh.jpg"
                                            caminho = os.path.join(PASTA_PLACAS, nome_arquivo)
                                            cv2.imwrite(caminho, recorte_melhorado)
                                            #print(f"[SALVO] Imagem da placa: {caminho}")
                                            rastreamento[cid]['placa_capturada'] = True
                        break
                
                #Se um carro nao foi rastreado, itera e cria um NOVO ID esperando o rastreamento de um novo frame processado pelo YOLO.
                if not carro_detectado:
                    id_carro += 1
                    rastreamento[id_carro] = {
                        'cx': cx,
                        'cy': cy,
                        'box': (x1, y1, x2, y2),
                        'detectado': True,
                        'frames_ausente': 0,
                        'ativo': True,
                        'label': label
                    }

            #limpa carros que nao aparecem mais.
            for cid in list(rastreamento.keys()):
                if not rastreamento[cid]['detectado']:
                    rastreamento[cid]['frames_ausente'] += 1
                    if rastreamento[cid]['frames_ausente'] > max_frames_ausente:
                        if 'velocidade' in rastreamento[cid]:
                            rastreamento[cid]['ativo'] = False
                        else:
                            del rastreamento[cid]

        #Desenha a bbox na tela em tempo de frame processado + pulado e retorna velocidade, id e tipo do carro.
        for cid, dados in rastreamento.items():
            if dados.get('ativo', True):
                x1, y1, x2, y2 = dados['box']
                if 'velocidade' in dados:
                    vel = dados['velocidade']
                    if vel < 40:
                        cor = (0, 255, 0)
                    elif vel < 80:
                        cor = (0, 255, 255)
                    else:
                        cor = (0, 0, 255)
                else:
                    cor = (255, 165, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                info = f"ID: {cid}"
                if 'label' in dados:
                    info += f" - {dados['label']}"
                if 'velocidade' in dados:
                    info += f" - {dados['velocidade']:.1f} km/h"
                cv2.putText(frame, info, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        #Reduz a contagem de frame e atualiza a cada 10 frames, possivelmente precisamos ajustar caso precise pular mais frames.
        if frame_count % 10 == 0:
            tempo_atual = time.time()
            fps_real = 10 / (tempo_atual - ultimo_fps) if tempo_atual > ultimo_fps else 0
            ultimo_fps = tempo_atual

        #Mostra na tela lógica de skip de processamento de frames do YOLO. Mostrando qual frame está sendo processado e qual está sendo skipado.
        cv2.putText(frame, f"FPS: {fps_real:.1f}", (largura - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        veiculos_ativos = sum(1 for v in rastreamento.values() if v.get('ativo', True))
        cv2.putText(frame, f"Veiculos: {veiculos_ativos}", (largura - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        status = "Processando" if eh_frame_processado else "Pulando"
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Detecção de Veículos", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Processamento finalizado")

if __name__ == "__main__":
    main()

import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

def detectar_trincas(imagem, confianca):
    if imagem is None:
        return None, "⚠️ Nenhuma imagem fornecida."
    img_bgr = cv2.cvtColor(np.array(imagem), cv2.COLOR_RGB2BGR)
    pred = model.predict(source=img_bgr, imgsz=640, conf=confianca, verbose=False)[0]
    img_result = cv2.cvtColor(pred.plot(), cv2.COLOR_BGR2RGB)
    n_det = len(pred.boxes) if pred.boxes is not None else 0
    if n_det == 0:
        status = "✅ NENHUMA TRINCA DETECTADA\n\nA parede analisada não apresenta fissuras visíveis dentro do limiar de confiança configurado.\nEstrutura aprovada para prosseguimento das etapas de revestimento."
    else:
        confs = pred.boxes.conf.cpu().numpy()
        status = (
            f"⚠️ ATENÇÃO — {n_det} TRINCA(S) DETECTADA(S)\n\n"
            f"Confiança média   : {confs.mean():.1%}\n"
            f"Confiança máxima  : {confs.max():.1%}\n"
            f"Confiança mínima  : {confs.min():.1%}\n\n"
            f"Recomendação: Realizar inspeção técnica presencial antes de\nprosseguir com revestimento ou pintura."
        )
    return img_result, status


css = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800;900&family=Oswald:wght@400;500;600;700&display=swap');

:root {
    --red:      #cc1f1f;
    --red-dark: #a01515;
    --dark:     #1a1a1a;
    --nav:      #222222;
    --white:    #ffffff;
    --gray-100: #f4f4f4;
    --gray-200: #e8e8e8;
    --gray-500: #777777;
    --border:   #dedede;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: var(--white) !important;
    font-family: 'Montserrat', sans-serif !important;
    color: var(--dark) !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* ── TOP NAV ── */
.top-nav {
    background: var(--nav);
    padding: 0 40px;
    display: flex;
    align-items: center;
    height: 48px;
    border-bottom: 3px solid var(--red);
}

.nav-item {
    color: #ffffff !important;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0 18px;
    height: 48px;
    display: flex;
    align-items: center;
    border-right: 1px solid rgba(255,255,255,0.1);
    cursor: default;
    opacity: 0.7;
}

.nav-item.active {
    background: var(--red);
    opacity: 1;
    color: #ffffff !important;
}

/* ── HEADER ── */
.site-header {
    background: var(--white);
    padding: 22px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

.logo-title {
    font-family: 'Oswald', sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: var(--dark) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    line-height: 1;
}

.logo-title span { color: var(--red) !important; }

.logo-tagline {
    font-size: 10px;
    color: var(--gray-500) !important;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 5px;
}

.header-badge {
    background: var(--red);
    color: #ffffff !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 8px 18px;
    display: inline-block;
}

.header-program {
    font-size: 11px;
    color: var(--gray-500) !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
    text-align: right;
    margin-bottom: 6px;
}

/* ── HERO ── */
.hero-banner {
    background: linear-gradient(90deg, #1a1a1a 0%, #2a2a2a 50%, #1f1f1f 100%);
    padding: 56px 40px 50px;
    position: relative;
    overflow: hidden;
    border-left: 6px solid var(--red);
}

.hero-arrow {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}

.hero-arrow-icon {
    width: 34px; height: 34px;
    background: var(--red);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    color: #ffffff !important;
    font-weight: 900;
    flex-shrink: 0;
}

.hero-arrow-label {
    font-size: 11px;
    color: #bbbbbb !important;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 700;
}

.hero-title {
    font-family: 'Oswald', sans-serif !important;
    font-size: clamp(32px, 4vw, 54px) !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    line-height: 1.1 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-bottom: 18px !important;
}

.hero-title span { color: var(--red) !important; }

.hero-desc {
    font-size: 14px;
    color: #dddddd !important;
    font-weight: 400;
    line-height: 1.8;
    max-width: 600px;
    margin-bottom: 38px;
}

.hero-stats { display: flex; gap: 0; flex-wrap: wrap; }

.stat-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-right: none;
    padding: 15px 24px;
    text-align: center;
    min-width: 110px;
}

.stat-box:last-child { border-right: 1px solid rgba(255,255,255,0.12); }

.stat-num {
    font-family: 'Oswald', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--red) !important;
    display: block;
}

.stat-lbl {
    font-size: 9px;
    color: #cccccc !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    margin-top: 4px;
    display: block;
}

/* ── MAIN AREA ── */
.main-wrap {
    padding: 44px 40px;
    background: var(--white);
}

.section-eyebrow {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--red) !important;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-eyebrow::before {
    content: '';
    display: inline-block;
    width: 20px; height: 2px;
    background: var(--red);
}

.section-heading {
    font-family: 'Oswald', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--dark) !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 22px;
    padding-bottom: 14px;
    border-bottom: 2px solid var(--red);
    display: inline-block;
}

/* ── CARD ── */
.card {
    border: 1px solid var(--border);
    background: var(--white);
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.card-head {
    background: var(--nav);
    padding: 11px 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-left: 4px solid var(--red);
}

.card-dot { width:7px; height:7px; background:var(--red); border-radius:50%; }

.card-head-title {
    font-family: 'Oswald', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #ffffff !important;
}

/* ── GRADIO ELEMENTS ── */
.upload-zone {
    background: var(--gray-100) !important;
    border: 2px dashed var(--gray-200) !important;
    border-radius: 0 !important;
    min-height: 260px !important;
    transition: all .3s !important;
}

.upload-zone:hover {
    border-color: var(--red) !important;
    background: rgba(204,31,31,0.03) !important;
}

.slider-wrap {
    padding: 18px 20px;
    background: var(--gray-100);
    border-top: 1px solid var(--border);
}

input[type=range] { accent-color: var(--red) !important; }

.analyze-btn {
    margin: 16px 20px !important;
    background: var(--red) !important;
    color: #ffffff !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 16px !important;
    width: calc(100% - 40px) !important;
    cursor: pointer !important;
    transition: all .2s !important;
}

.analyze-btn:hover {
    background: var(--red-dark) !important;
    box-shadow: 0 6px 20px rgba(204,31,31,0.3) !important;
    transform: translateY(-1px) !important;
}

.result-image { min-height: 260px !important; background: #111 !important; }

.report-area textarea {
    background: var(--gray-100) !important;
    border: none !important;
    border-top: 1px solid var(--border) !important;
    color: var(--dark) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1.8 !important;
    padding: 20px !important;
    border-radius: 0 !important;
}

/* ── FOOTER ── */
.site-footer {
    background: var(--nav);
    padding: 28px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    border-top: 3px solid var(--red);
}

.footer-left {
    font-size: 13px;
    color: #ffffff !important;
    font-weight: 400;
    letter-spacing: 0.3px;
}

.footer-left strong { color: #ffffff !important; font-weight: 700; }
.footer-left em { color: var(--red) !important; font-style: normal; font-weight: 600; }

.footer-tags { display: flex; gap: 8px; flex-wrap: wrap; }

.ftag {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: #ffffff !important;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 5px 12px;
}

/* ── OVERRIDES ── */
.gr-box, .gr-form, .gr-panel { background: transparent !important; border: none !important; }
label { color: #444444 !important; font-size: 11px !important; font-weight: 700 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
footer { display: none !important; }
"""

HEADER_HTML = """
<div class="top-nav">
    <div class="nav-item active">Início</div>
    <div class="nav-item">Sobre o Sistema</div>
    <div class="nav-item">Detecção de Fissuras</div>
    <div class="nav-item">Resultados</div>
    <div class="nav-item">Contato</div>
</div>
<div class="site-header">
    <div>
        <div class="logo-title">INSPECT<span>IA</span></div>
        <div class="logo-tagline">Detecção Inteligente de Fissuras Estruturais</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px;">
        <div class="header-program">Programa de Residência em IA · UNISENAI</div>
        <div class="header-badge">Construção Civil · IA · YOLOv8</div>
    </div>
</div>
<div class="hero-banner">
    <div class="hero-arrow">
        <div class="hero-arrow-icon">→</div>
        <div class="hero-arrow-label">Visão Computacional Aplicada à Engenharia Civil</div>
    </div>
    <h1 class="hero-title">
        Detecção Automática de<br>
        <span>Trincas e Fissuras</span> em Paredes
    </h1>
    <p class="hero-desc">
        Solução embarcada de inteligência artificial para inspeção automatizada de paredes em obras civis.
        Identifica e localiza fissuras estruturais com precisão, reduzindo retrabalho e garantindo
        a qualidade do revestimento antes da entrega do empreendimento.
    </p>
    <div class="hero-stats">
        <div class="stat-box"><span class="stat-num">73.3%</span><span class="stat-lbl">mAP@50</span></div>
        <div class="stat-box"><span class="stat-num">76.4%</span><span class="stat-lbl">Precisão</span></div>
        <div class="stat-box"><span class="stat-num">4.8ms</span><span class="stat-lbl">Inferência</span></div>
        <div class="stat-box"><span class="stat-num">6.8 MB</span><span class="stat-lbl">Modelo</span></div>
        <div class="stat-box"><span class="stat-num">1.551</span><span class="stat-lbl">Amostras</span></div>
    </div>
</div>
"""

FOOTER_HTML = """
<div class="site-footer">
    <div class="footer-left">
        <strong>InspectIA</strong> &nbsp;·&nbsp;
        Programa de Residência em Inteligência Artificial &nbsp;·&nbsp;
        UNISENAI 2026 &nbsp;·&nbsp;
        <em>Daniel Tavares de França</em>
    </div>
    <div class="footer-tags">
        <span class="ftag">YOLOv8n-seg</span>
        <span class="ftag">Instance Segmentation</span>
        <span class="ftag">Computer Vision</span>
        <span class="ftag">Construção Civil</span>
    </div>
</div>
"""

with gr.Blocks(css=css, title="InspectIA — Detector de Trincas") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row(elem_classes=["main-wrap"], equal_height=True):
        with gr.Column():
            gr.HTML('<div class="section-eyebrow">Entrada</div><div class="section-heading">Imagem para Análise</div><div class="card"><div class="card-head"><div class="card-dot"></div><span class="card-head-title">Upload da Imagem</span></div>')
            img_input = gr.Image(type="pil", label="", show_label=False, elem_classes=["upload-zone"])
            confianca = gr.Slider(minimum=0.10, maximum=0.90, value=0.40, step=0.05, label="Limiar de Confiança", elem_classes=["slider-wrap"])
            btn = gr.Button("→  Iniciar Análise Estrutural", variant="primary", elem_classes=["analyze-btn"])
            gr.HTML("</div>")

        with gr.Column():
            gr.HTML('<div class="section-eyebrow">Resultado</div><div class="section-heading">Detecção de Fissuras</div><div class="card"><div class="card-head"><div class="card-dot"></div><span class="card-head-title">Segmentação de Instância</span></div>')
            img_output = gr.Image(type="numpy", label="", show_label=False, elem_classes=["result-image"])
            gr.HTML('<div class="card-head" style="border-top:1px solid #333"><div class="card-dot"></div><span class="card-head-title">Laudo Técnico Automatizado</span></div>')
            txt_output = gr.Textbox(label="", lines=5, show_label=False, elem_classes=["report-area"])
            gr.HTML("</div>")

    gr.HTML(FOOTER_HTML)

    btn.click(fn=detectar_trincas, inputs=[img_input, confianca], outputs=[img_output, txt_output])

demo.launch()

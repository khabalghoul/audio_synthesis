# Síntesis audiorítmica en tiempo real con StyleGAN 3

Este proyecto implementa un pipeline audiovisual que convierte audio en tiempo real proveniente de un sintetizador en imágenes generadas mediante StyleGAN3. El flujo del sistema es el siguiente:

```
Audio (JACK) → audio_processor.py → Vector latente → server.py → visualizer.py → Imagen generada (StyleGAN3)
```

---

## 🚧 Requisitos previos

- **Hardware**: GPU NVIDIA compatible con CUDA ≥ 11.3  
- **Software de audio**: JACK (servidor de audio en tiempo real, por ejemplo usando `qjackctl`)
- **StyleGAN3**: https://github.com/NVlabs/stylegan3 (no es necesario pero si necesitamos poder correr el codigo de stylegan con todas sus dependencias, es la parte dificil)

**Librerías Python requeridas**:

```bash
pip install numpy librosa jack-client torch torchvision torchaudio imgui matplotlib dnnlib
```

Se recomienda configurar StyleGAN3 previamente en un entorno virtual:


---

## 📂 Estructura del repositorio

```
.
├── audio_processor.py   # Captura de audio y generación del vector latente
├── server.py            # Reenvío del vector latente vía TCP
├── visualizer.py        # Interfaz visual que integra StyleGAN3
```

---

## Ejecución del sistema

Ejecutar cada script en una terminal independiente, en el siguiente orden:

1. **Captura y procesamiento del audio**:

```bash
python audio_processor.py
```

2. **Servidor intermediario (reenviador TCP)**:

```bash
python server.py
```

3. **Visualizador  StyleGAN3**:

```bash
python visualizer.py
```

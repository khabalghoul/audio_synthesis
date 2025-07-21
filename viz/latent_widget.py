# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import socket
import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils
import json

import threading  # ← NUEVO
import queue

# ----------------------------------------------------------------------------
HOST = 'localhost'
PORT = 65437


class LatentWidget:
    def __init__(self, viz):
        self.viz = viz
        self.latent = dnnlib.EasyDict(x=0, y=0, anim=False, speed=0.25)
        self.dz = np.zeros(512, dtype=np.float32)  # ← NUEVO
        self.latent_def = dnnlib.EasyDict(self.latent)
        self.step_y = 100
        self.sensitivity_factor = 0.01

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))
        self.s.listen()

        self.s.setblocking(False)  # <── ①  no bloquear accept()
        self._conn = None
        # print("Connected to server at", (HOST, PORT))

        self._z_queue = queue.Queue(maxsize=1)  # solo guardamos el último vector
        threading.Thread(target=self._reader_thread,  # hilo “daemon” → muere con la app
                         daemon=True).start()

    def drag(self, dx, dy):
        viz = self.viz
        self.latent.x += dx / viz.font_size * 4e-2
        self.latent.y += dy / viz.font_size * 4e-2

    def _reader_thread(self):
        buf = b''  # fragmentos pendientes
        SZ = 512 * 4  # bytes por vector
        while True:
            try:
                # 1) aceptar conexión (bloqueante):
                if self._conn is None:
                    self._conn, _ = self.s.accept()

                # 2) leer del socket:
                data = self._conn.recv(4096)
                if not data:  # remoto cerró
                    self._conn.close()
                    self._conn = None
                    continue  # volver a aceptar

                buf += data

                # 3) procesar todos los vectores completos:
                while len(buf) >= SZ:
                    raw, buf = buf[:SZ], buf[SZ:]
                    try:  # pisamos la cola (máx. 1)
                        self._z_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._z_queue.put_nowait(raw)

            except (BlockingIOError, ConnectionResetError, OSError):
                continue  # vuelve al while y reintenta

    def fetch_translation_from_server(self):
        # ← Este método se llama cada frame desde el hilo GUI
        try:
            raw = self._z_queue.get_nowait()  # no bloquea
        except queue.Empty:
            return  # nada nuevo

        v = np.frombuffer(raw, np.float32)

        # low-pass opcional para no “temblar” en cada frame
        self._z_prev = getattr(self, "_z_prev",
                               np.zeros_like(v))
        v = 0.85 * self._z_prev + 0.15 * v
        self._z_prev = v

        # se pasa como bytes al renderer
        self.viz.args.z_offset = v.tobytes()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        self.fetch_translation_from_server()
        self._lp_pos = getattr(self, "_lp_pos", np.array([self.latent.x, self.latent.y], np.float32))

        viz = self.viz
        if show:
            # print('latent X', self.latent.x)
            # print('latent Y', self.latent.y)
            imgui.text('Latent')
            imgui.same_line(viz.label_w)
            seed = round(self.latent.x) + round(self.latent.y) * self.step_y
            with imgui_utils.item_width(viz.font_size * 8):
                changed, seed = imgui.input_int('##seed', seed)
                if changed:
                    self.latent.x = seed
                    self.latent.y = 0
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            frac_x = self.latent.x - round(self.latent.x)
            frac_y = self.latent.y - round(self.latent.y)
            with imgui_utils.item_width(viz.font_size * 5):
                changed, (new_frac_x, new_frac_y) = imgui.input_float2('##frac', frac_x, frac_y, format='%+.2f',
                                                                       flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if changed:
                    self.latent.x += new_frac_x - frac_x
                    self.latent.y += new_frac_y - frac_y
            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.spacing * 2)
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=viz.button_w)
            if dragging:
                self.drag(dx, dy)
            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.button_w + viz.spacing * 3)
            _clicked, self.latent.anim = imgui.checkbox('Anim', self.latent.anim)
            imgui.same_line(round(viz.font_size * 27.7))
            with imgui_utils.item_width(-1 - viz.button_w * 2 - viz.spacing * 2), imgui_utils.grayed_out(
                    not self.latent.anim):
                changed, speed = imgui.slider_float('##speed', self.latent.speed, -5, 5, format='Speed %.3f', power=3)
                if changed:
                    self.latent.speed = speed
            imgui.same_line()
            snapped = dnnlib.EasyDict(self.latent, x=round(self.latent.x), y=round(self.latent.y))
            if imgui_utils.button('Snap', width=viz.button_w, enabled=(self.latent != snapped)):
                self.latent = snapped
            imgui.same_line()
            if imgui_utils.button('Reset', width=-1, enabled=(self.latent != self.latent_def)):
                self.latent = dnnlib.EasyDict(self.latent_def)

        if self.latent.anim:
            self.latent.x += viz.frame_delta * self.latent.speed
        viz.args.w0_seeds = []  # [[seed, weight], ...]
        for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            ALPHA = 0.15  # 0=sin filtro, 1=sigue igual que ahora
            self._lp_pos = (1 - ALPHA) * self._lp_pos + ALPHA * np.array([self.latent.x, self.latent.y])
            x_smooth, y_smooth = self._lp_pos
            seed_x = np.floor(x_smooth) + ofs_x
            seed_y = np.floor(y_smooth) + ofs_y
            seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
            weight = (1 - abs(self.latent.x - seed_x)) * (1 - abs(self.latent.y - seed_y))
            # fx = abs(x_smooth - seed_x)
            # fy = abs(y_smooth - seed_y)
            # if fx > 1 or fy > 1:
            #     continue  # fuera de la celda 2×2
            # # smoothstep: S(t)=3t²-2t³ ; invertimos porque queremos 1 en el centro, 0 en el borde
            # sx = 1 - (3 * fx * fx - 2 * fx * fx * fx)
            # sy = 1 - (3 * fy * fy - 2 * fy * fy * fy)
            # weight = sx * sy
            if weight > 0:
                viz.args.w0_seeds.append([seed, weight])

# ----------------------------------------------------------------------------

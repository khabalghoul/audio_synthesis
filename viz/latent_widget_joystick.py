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

import threading          # ← NUEVO
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

    def fetch_translation_from_server____(self):
        # 1) aceptar conexión (no bloqueante)
        if self._conn is None:
            try:
                self._conn, _ = self.s.accept()
                self._conn.setblocking(False)
            except BlockingIOError:
                return

        # 2) recibir datos (no bloqueante)
        try:
            data = self._conn.recv(4096)
            if not data:  # conexión remota cerrada
                self._conn.close()
                self._conn = None
                return
        except BlockingIOError:  # nada nuevo en este frame
            return

        # ───────── a partir de aquí SÍ existe `data` ─────────
        self._buffer = getattr(self, "_buffer", b"") + data
        SZ = 512 * 4  # bytes por vector

        # nos quedamos con el vector más reciente y descartamos resto
        if len(self._buffer) >= SZ:
            keep = (len(self._buffer) // SZ) * SZ  # múltiplo completo
            last_raw = self._buffer[keep - SZ: keep]
            self._buffer = self._buffer[keep:]  # sobrante incompleto

            v = np.frombuffer(last_raw, np.float32)  # (512,)

            # opcional: pequeño low-pass para suavizar
            self._z_prev = getattr(self, "_z_prev", np.zeros(512, np.float32))
            v = 0.85 * self._z_prev + 0.15 * v
            self._z_prev = v

            self.viz.args.z_offset = v.tobytes()

    def fetch_translation_from_server___(self):
        # 1) aceptar conexión (no bloqueante)
        if self._conn is None:
            try:
                self._conn, _ = self.s.accept()
                self._conn.setblocking(False)
            except BlockingIOError:
                return

        # 2) leer vector (2 KiB) — puede venir pegado, así que usamos while
        try:
            data = self._conn.recv(4096)
            if not data:
                self._conn.close()
                self._conn = None
                return
        except BlockingIOError:
            return

        self._buffer = getattr(self, "_buffer", b"") + data
        SZ = 512 * 4  # bytes por vector
        while len(self._buffer) >= SZ:
            raw, self._buffer = self._buffer[:SZ], self._buffer[SZ:]
            v = np.frombuffer(raw, np.float32)  # shape (512,)
            #self.viz.args.z_offset = v  # ← pasa al renderer
            #self.viz.args.z_offset = v.tobytes()
            self._z_prev = getattr(self, "_z_prev", np.zeros(512, np.float32))
            v = 0.85 * self._z_prev + 0.15 * v
            self._z_prev = v
            self.viz.args.z_offset = v.tobytes()

    def fetch_translation_from_server__(self):
        # 1) aceptar nueva conexión si no tenemos
        if self._conn is None:
            try:
                self._conn, _ = self.s.accept()
                self._conn.setblocking(False)
            except BlockingIOError:
                return  # nada que leer todavía

        # 2) recibir datos (no bloqueante)
        try:
            data = self._conn.recv(1024)
            if not data:  # conexión cerrada desde el otro lado
                self._conn.close()
                self._conn = None
                return
        except BlockingIOError:
            return  # no llegó nada en este frame

        # 3) procesar uno o varios mensajes
        for msg in data.decode().split('\n'):
            if not msg:
                continue
            try:
                d = json.loads(msg)
            except json.JSONDecodeError:
                continue

            # ---- aquí tu lógica original -------------
            if 'dx' in d: self.latent.x += float(d['dx'])
            if 'dy' in d: self.latent.y += float(d['dy'])
            if "dz" in d:
                # v = np.asarray(d["dz"], dtype=np.float32)
                v = float(d['dz'])
                print('dz recibido', v)

                # if v.size == 512:
                #     self.dz += v  # acumulamos Δ
                #     self.viz.args.z_offset = self.dz  # pasa al renderer

            continue

    def fetch_translation_from_server_(self):
        conn, _ = self.s.accept()
        with conn:
            data = conn.recv(1024)
            if data:
                messages = data.decode('utf-8').split('\n')
                for message in messages:
                    if message:  # Ignore empty messages
                        data_dict = json.loads(message)
                        if 'dx' in data_dict:  # incremento ejes
                            self.latent.x += float(data_dict['dx'])
                        if 'dy' in data_dict:
                            self.latent.y += float(data_dict['dy'])

                        if 'R2' in data_dict and data_dict['R2'] is not None:
                            # Increase animation speed
                            self.latent.speed += float(data_dict['R2']) * self.sensitivity_factor
                        if 'L2' in data_dict and data_dict['L2'] is not None:
                            # Decrease animation speed
                            self.latent.speed -= float(data_dict['L2']) * self.sensitivity_factor
                        if 'triangle' in data_dict and int(data_dict['triangle']) == 1:
                            print('triangle is not none')
                            #    # Decrease animation speed
                            self.latent.speed *= -1
                        if 'start' in data_dict and data_dict['start']:
                            self.latent.x += self.sensitivity_factor  # Adjust as needed
                        if 'select' in data_dict and data_dict['select']:
                            self.latent.y += self.sensitivity_factor  # Adjust as needed
                        if 'square' in data_dict and data_dict['square'] is not None:
                            self.latent.anim = bool(data_dict['square'])
                            if data_dict['square'] == 1:
                                if 'R1' in data_dict and data_dict['R1']:
                                    self.latent.x += self.sensitivity_factor  # Adjust as needed
                                if 'L1' in data_dict and data_dict['L1']:
                                    self.latent.y += self.sensitivity_factor  # Adjust as needed

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

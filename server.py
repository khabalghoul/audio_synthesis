# server2.py  — reenviar solo a LatentWidget, sin caerse
import socket, threading

HOST, PORT = "localhost", 65431
LATENT_HOST, LATENT_PORT = "localhost", 65437
BUF = 4096


def fwd_to_latent_(pkg: bytes):
    """Reenvía; si el widget todavía no está, sólo avisa."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((LATENT_HOST, LATENT_PORT))
            s.sendall(pkg)
    except ConnectionRefusedError:
        # el visualizador aún no levantó el puerto
        pass
    except Exception as e:
        print("⚠️  al reenviar:", e)


LAT_SOCK = None


def fwd_to_latent(pkg):
    global LAT_SOCK
    if LAT_SOCK is None:
        LAT_SOCK = socket.socket()
        try:
            LAT_SOCK.connect((LATENT_HOST, LATENT_PORT))
        except ConnectionRefusedError:
            LAT_SOCK = None
            return
    try:
        LAT_SOCK.sendall(pkg)
    except OSError:  # widget se cerró
        LAT_SOCK = None


def handle(cli):
    with cli:
        while True:
            data = cli.recv(BUF)
            if not data:
                break
            print("srv ⮕", data)
            fwd_to_latent(data)


def main():
    print("🛰  server escuchando", HOST, PORT)
    with socket.socket() as srv:
        srv.bind((HOST, PORT))
        srv.listen()
        while True:
            cli, _ = srv.accept()
            threading.Thread(target=handle, args=(cli,), daemon=True).start()


if __name__ == "__main__":
    main()

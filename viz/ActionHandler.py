import socket


class ActionHandler:
    def set_socket(self, host, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((host, port))
        self.s.listen()
        print("Connected to server at", (host, port))

    def fetch_translation_from_server(self):
        raise NotImplementedError

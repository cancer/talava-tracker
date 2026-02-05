from pythonosc import udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 39570)

# トラッカー0を有効化、位置(0, 1, 0)、回転なし
client.send_message("/VMT/Room/Unity", [0, 1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

print("sent!")

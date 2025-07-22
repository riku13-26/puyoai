import pypuyo as ppy

game = ppy.start(width=6, height=12)  # 6×12 のフィールド
for _ in range(5):
    game.update()
    print(game.get())  # 2 次元リストで盤面が返る

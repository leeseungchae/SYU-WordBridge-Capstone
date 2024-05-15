from kotan import KoTAN


mt = KoTAN(task="translation", tgt="en")
inputs = ['나는 온 세상 사람들이 행복해지길 바라', '나는 선한 영향력을 펼치는 사람이 되고 싶어']
mt.predict(inputs)
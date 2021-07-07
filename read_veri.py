import numpy as np
veri = open("veri_test.txt")
f = veri.readlines()
veri_label = []
veri_enroll = []
veri_test = []

for line in f:
    words = line.split()
    veri_label.append(int(words[0]))
    veri_enroll.append("./ver/"+words[1])
    veri_test.append("./ver/"+words[2])

np.save('veri_label', np.asarray(veri_label))
np.save('veri_test', np.asarray(veri_test))
np.save('veri_enroll', np.asarray(veri_enroll))
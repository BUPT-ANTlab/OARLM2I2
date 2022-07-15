from predict import MVP_Informer
pr = MVP_Informer()
for i in range(1,20):
    his=[i for j in range(48)]
    pr.save_his(his)

print(len(pr.pre()[0]))


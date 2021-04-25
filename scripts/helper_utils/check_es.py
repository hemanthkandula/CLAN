from Project.UMAAnet.scripts.helper_utils.EarlyStopping import EarlyStopping

early_stopping = EarlyStopping(patience=10, verbose=True,delta=.01)
k =10

for i in range(1000):
    early_stopping(k)
    print(i)
    k=k-0.0001

    if early_stopping.early_stop:
        print("Early stopping")
        break
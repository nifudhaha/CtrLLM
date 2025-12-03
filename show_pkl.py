import pickle

with open("demo_metrics.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data)

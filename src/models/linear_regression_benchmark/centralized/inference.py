import skops.io as sio


model = sio.load('linear_regression_benchmark.skops', trusted=True)
model.predict(features)

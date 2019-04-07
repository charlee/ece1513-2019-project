python main.py --model lr --path __model-lr__ --epoch=400
python main.py --model nn --path __model-nn__ --epoch=400
python main.py --model cnn --path __model-cnn__ --epoch=400
python main.py --model simplenet --path __model-simplenet__ --epoch=400

python plot.py

python main.py --model lr --path __model-lr__ --predict
python main.py --model nn --path __model-nn__ --predict
python main.py --model cnn --path __model-cnn__ --predict
python main.py --model simplenet --path __model-simplenet__ --predict
